from __future__ import annotations

import json
import logging
import re

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

from mta.agents.base import Action, BaseAgent, Step, Trajectory
from mta.agents.system_prompts import (
    SWEAGENT_SYSTEM_PROMPT,
    SWEAGENT_USER_PROMPT,
    SWE_SYSTEM_PROMPT,
    SWE_SYSTEM_PROMPT_FN_CALL,
    SWE_USER_PROMPT,
    SWE_USER_PROMPT_FN_CALL,
)

TOKEN_WARNING_THRESHOLD = 28000

logger = logging.getLogger(__name__)


def parse_oai_response(response):
    thought = response.choices[0].message.content
    if not thought:
        thought = ""
    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        action = SWEAction(function_name, parameters)
    except Exception:
        action = SWEAction(function_name="", parameters={})
    return thought, action


def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)
        thought = response_text[: match.start()]
    else:
        thought = response_text
        action = ""

    thought = thought.strip()
    action = action.strip()

    action = SWEAction.from_string(action)

    return thought, action


class SWEAgent(BaseAgent):
    def __init__(self, use_fn_calling: bool = False, format_model_response: bool = False, scaffold: str = "r2egym"):
        if SWEAction is None:
            raise ImportError("r2egym is required to use SWEAgent; install it via `pip install mta[env]`.")
        self.use_fn_calling = use_fn_calling
        self.format_model_response = format_model_response
        self.scaffold = scaffold
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        self.system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if use_fn_calling else SWE_SYSTEM_PROMPT
        if scaffold == "sweagent":
            self.system_prompt = SWEAGENT_SYSTEM_PROMPT
        self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT

        self._trajectory = Trajectory()
        self.reset()

    def process_model_response(self, response: str) -> tuple[str, str]:
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            thought, action = parse_xml_response(response)

        action_str = action.to_xml_string()
        if self.format_model_response:
            response = f"{thought}\n\n{action_str}"
        return action.to_xml_string(), {
            "thought": thought,
        }

    def update_from_env(self, observation, reward, done, info):
        if self._trajectory.steps:
            observation = str(observation)
        else:
            observation = str(observation)
            observation = self.user_prompt_template.format(problem_statement=observation)

        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."

        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.next_observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info

        self.messages.append({"role": "user", "content": observation})
        self.cur_step = Step(observation=observation)

    def update_from_model(self, response: str, **kwargs):
        self._trajectory.steps.append(self.cur_step)
        if self.use_fn_calling:
            thought, action = parse_oai_response(response)
        else:
            thought, action = parse_xml_response(response)
        action_str = action.to_xml_string()
        assert self._trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        cur_step = self._trajectory.steps[-1]
        cur_step.thought = thought
        cur_step.action = action_str
        cur_step.model_response = response

        if self.format_model_response:
            self.messages.append({"role": "assistant", "content": f"{thought}\n\n{action_str}"})
        else:
            self.messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)

    def get_current_state(self) -> Step:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self.step = 0

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
