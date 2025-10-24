import asyncio
import concurrent.futures
import json
import logging
import time
from pathlib import Path

import openai
import torch
from openai.types import Completion
from openai.types.chat import ChatCompletion
try:
    from openai.types.responses import Response
except ImportError:  # pragma: no cover - fallback for older SDKs
    Response = None  # type: ignore[assignment]

from mta.agents.base import Action, BaseAgent, Trajectory
from mta.agents.utils import convert_messages_to_tokens_and_masks, get_recent_assistant_user_messages
from mta.environments.env_utils import compute_mc_return, compute_trajectory_reward
from mta.misc import colorful_print
from mta.parser.chat_template.parser import ChatTemplateParser

logger = logging.getLogger(__name__)


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=1,
        trajectory_timeout=None,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,
        enforce_max_prompt_length=False,
        overlong_filter=False,
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.overlong_filter = overlong_filter

        self.gamma = gamma
        self.retry_limit = retry_limit
        self.api_retries = api_retries
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        self.openai_completion_limit = kwargs.get("openai_completion_limit")

        self.agent_class = agent_class
        self.agent_args = agent_args
        if isinstance(self.agent_args, dict):
            self.return_raw_openai_response = bool(self.agent_args.get("use_fn_calling"))
        else:
            self.return_raw_openai_response = False
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"
        self.rollout_engine_args = rollout_engine_args or {}
        self.sampling_params = kwargs.get("sampling_params", {})

        self.client = None
        self.local_model = None
        self.generation_device = None

        self.write_step_logs = kwargs.get("write_step_logs", True)
        logs_dir = kwargs.get("step_logs_dir", "trajectory_logs")
        self.step_logs_dir = Path(logs_dir) if self.write_step_logs else None
        if self.write_step_logs and self.step_logs_dir is not None:
            self.step_logs_dir.mkdir(parents=True, exist_ok=True)

        if self.engine_name == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(**self.rollout_engine_args)
            logging.getLogger("httpx").setLevel(logging.WARNING)
        elif self.engine_name == "transformers":
            assert rollout_engine is not None, "A local transformers model must be provided when engine_name='transformers'"
            self.local_model = rollout_engine
            device_opt = self.rollout_engine_args.get("device")
            if device_opt and device_opt not in ("auto",):
                self.generation_device = torch.device(device_opt)
            else:
                try:
                    self.generation_device = next(self.local_model.parameters()).device
                except (StopIteration, AttributeError):
                    self.generation_device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported engine_name: {self.engine_name}")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
        else:
            self.chat_parser = chat_parser

    async def get_model_response(self, prompt, application_id, **kwargs):
        if self.engine_name == "openai":
            return await self._get_openai_async(prompt, application_id, **kwargs)
        if self.engine_name == "transformers":
            return await self._get_transformers_async(prompt, application_id, **kwargs)
        raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents
        self.n_parallel_agents = len(envs)

    async def _get_openai_async(self, prompt, _, **kwargs):
        is_chat_prompt = isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt)
        retries = self.api_retries
        last_error: Exception | None = None

        disallow_max_tokens = False

        while retries > 0:
            request_params = dict(self.sampling_params)
            request_params.update(kwargs)
            request_params["timeout"] = 3600

            if not disallow_max_tokens:
                max_tokens = request_params.get("max_tokens")
                if self.openai_completion_limit is not None and max_tokens is not None:
                    if max_tokens > self.openai_completion_limit:
                        logger.warning(
                            "Clamping max_tokens from %s to service limit %s",
                            max_tokens,
                            self.openai_completion_limit,
                        )
                    clamped = min(max_tokens, self.openai_completion_limit)
                    request_params["max_tokens"] = clamped
                    request_params["max_completion_tokens"] = clamped
            else:
                request_params.pop("max_tokens", None)
                if self.openai_completion_limit is not None:
                    request_params.setdefault("max_completion_tokens", self.openai_completion_limit)

            try:
                response = await self._attempt_openai_chat_or_completion(prompt, request_params, is_chat_prompt)
                return self._format_openai_response(response)
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                logger.info("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                if self._is_max_tokens_unsupported(exc) and not disallow_max_tokens:
                    disallow_max_tokens = True
                    continue
                if Response is not None and self._should_fallback_to_responses(exc):
                    try:
                        response = await self._call_openai_responses(prompt, request_params, is_chat_prompt)
                        return self._format_openai_response(response)
                    except Exception as resp_exc:  # pragma: no cover - network dependency
                        logger.error("Responses API fallback failed: %s", resp_exc)
                        return f"Error processing content: {resp_exc}"
                logger.error("Error: %s", exc)
                return f"Error processing content: {exc}"

        if last_error is not None:
            return f"Error processing content: {last_error}"
        return "Error: Unable to obtain response from OpenAI."

    async def _attempt_openai_chat_or_completion(self, prompt, request_params, is_chat_prompt):
        params = request_params.copy()
        if is_chat_prompt:
            params.pop("prompt", None)
            params["messages"] = prompt
            return await self.client.chat.completions.create(**params)

        prompt_text = prompt
        if isinstance(prompt, list):
            prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)
        params["prompt"] = prompt_text
        return await self.client.completions.create(**params)

    async def _call_openai_responses(self, prompt, request_params, is_chat_prompt):
        if Response is None:  # pragma: no cover - compatibility fallback
            raise RuntimeError("OpenAI responses API is not available in the installed openai package.")

        params = request_params.copy()
        max_tokens = params.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = params.pop("max_completion_tokens", None)
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        params.pop("prompt", None)
        params.pop("messages", None)

        if is_chat_prompt and isinstance(prompt, list):
            responses_input = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in prompt
            ]
        else:
            if isinstance(prompt, list):
                prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)
            else:
                prompt_text = prompt
            responses_input = [{"role": "user", "content": prompt_text}]

        params["input"] = responses_input
        return await self.client.responses.create(**params)

    def _should_fallback_to_responses(self, exc: Exception) -> bool:
        message = str(exc).lower()
        fallback_markers = [
            "not supported in the v1/completions endpoint",
            "use the responses api",
            "unknown parameter",
            "only supported in v1/responses",
        ]
        return any(marker in message for marker in fallback_markers)

    def _is_max_tokens_unsupported(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "use 'max_completion_tokens'" in message or "unsupported parameter: 'max_tokens'" in message

    def _format_openai_response(self, response):
        if isinstance(response, Completion):
            return response.choices[0].text
        if isinstance(response, ChatCompletion):
            if self.return_raw_openai_response:
                return response
            return response.choices[0].message.content or ""
        if Response is not None and isinstance(response, Response):
            if self.return_raw_openai_response:
                logger.warning("Responses API does not support returning raw ChatCompletion; returning text output instead.")
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text

            text_chunks: list[str] = []
            output = getattr(response, "output", None) or []
            for block in output:
                content_items = getattr(block, "content", []) or []
                for item in content_items:
                    text_obj = getattr(item, "text", None)
                    if text_obj is None:
                        continue
                    if isinstance(text_obj, str):
                        text_chunks.append(text_obj)
                    else:
                        value = getattr(text_obj, "value", None)
                        if value:
                            text_chunks.append(value)
            return "\n".join(text_chunks)
        return response

    def _get_response_display_text(self, response):
        if isinstance(response, ChatCompletion):
            try:
                message = response.choices[0].message
            except (AttributeError, IndexError):
                return str(response)
            text = getattr(message, "content", "") or ""
            if text:
                return text
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                tool_call = tool_calls[0]
                function_obj = getattr(tool_call, "function", None)
                name = getattr(function_obj, "name", "") if function_obj else ""
                arguments = getattr(function_obj, "arguments", "") if function_obj else ""
                return f"<function={name}>\n{arguments}\n</function>"
            return ""
        if isinstance(response, Completion):
            try:
                return response.choices[0].text
            except (AttributeError, IndexError):
                return str(response)
        if Response is not None and isinstance(response, Response):
            return self._format_openai_response(response)
        return response if isinstance(response, str) else str(response)

    async def _get_transformers_async(self, prompt, _=None, **kwargs):
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)
        else:
            prompt_text = prompt

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.sampling_params.get("max_tokens", self.max_response_length)

        temperature = kwargs.get("temperature", self.sampling_params.get("temperature", 1.0))
        top_p = kwargs.get("top_p", self.sampling_params.get("top_p", 1.0))
        do_sample = temperature is None or temperature > 0

        loop = asyncio.get_event_loop()

        def generate_text():
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            if self.generation_device is not None:
                inputs = {k: v.to(self.generation_device) for k, v in inputs.items()}

            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
            }
            if temperature is not None:
                generation_kwargs["temperature"] = temperature
            if top_p is not None and do_sample:
                generation_kwargs["top_p"] = top_p

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")

            with torch.no_grad():
                output_ids = self.local_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            generated_ids = output_ids[:, input_ids.shape[-1] :]
            text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return text

        return await loop.run_in_executor(self.executor, generate_text)

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        agent = self.agents[idx]
        env = self.envs[idx]

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        episode_steps = []
        step_logs = [] if self.write_step_logs else None

        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        agent.reset()
        agent.update_from_env(
            observation=observation,
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True,
        )
        prompt_token_len = len(prompt_tokens)
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            prompt_messages = agent.chat_completions.copy()

            env_message_content = None
            env_message_str = None
            for message in reversed(prompt_messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    env_message_content = message.get("content")
                    break
            if env_message_content is not None:
                env_message_str = str(env_message_content)
                colorful_print(f"[Trajectory {idx} | Step {step_idx}] Environment Input:\n{env_message_str}", "magenta")

            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens
            if self.engine_name == "openai" and self.openai_completion_limit is not None:
                if kwargs["max_tokens"] > self.openai_completion_limit:
                    logger.warning(
                        "Clamping max_tokens from %s to service limit %s",
                        kwargs["max_tokens"],
                        self.openai_completion_limit,
                    )
                kwargs["max_tokens"] = min(kwargs["max_tokens"], self.openai_completion_limit)

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            response_display = self._get_response_display_text(response)
            colorful_print(f"[Trajectory {idx} | Step {step_idx}] LLM Output:\n{response_display}", "blue")
            total_time += delta_time
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response_display,
            }
            episode_steps.append(prompt_response_pair)

            action: Action = agent.update_from_model(response)
            action = action.action
            current_step = agent.get_current_state()
            thought_str = str(getattr(current_step, "thought", "")) if current_step else ""
            action_str = str(getattr(current_step, "action", "")) if current_step else ""
            colorful_print(f"[Trajectory {idx} | Step {step_idx}] Thought:\n{thought_str}", "cyan")
            colorful_print(f"[Trajectory {idx} | Step {step_idx}] Action:\n{action_str}", "green")

            if step_logs is not None:
                step_logs.append(
                    {
                        "step": step_idx,
                        "environment_input": env_message_str,
                        "llm_output": response_display,
                        "thought": thought_str,
                        "action": action_str,
                    }
                )

            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=(self.trajectory_timeout - total_time),
                )
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be impossible"

            assistant_msg_tokens = []
            assistant_msg_masks = []
            if assistant_message is not None:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
                    [assistant_message],
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                )

            env_msg_tokens = []
            env_msg_masks = []
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser)

            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)

            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                truncation_length = self.max_response_length - response_token_len
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                break

            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason in {"TRUNCATION", "MAX_STEPS", "TIMEOUT"}:
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        await loop.run_in_executor(self.executor, env.close)

        if step_logs is not None:
            log_payload = {
                "trajectory_index": idx,
                "application_id": str(application_id),
                "termination_reason": termination_reason,
                "reward": reward,
                "steps": step_logs,
            }
            log_filename = self.step_logs_dir / f"trajectory_{application_id}.json" if self.step_logs_dir is not None else Path(f"trajectory_{application_id}.json")
            try:
                with log_filename.open("w", encoding="utf-8") as log_file:
                    json.dump(log_payload, log_file, ensure_ascii=False, indent=2)
            except Exception as exc:
                colorful_print(f"Failed to write trajectory log for {application_id}: {exc}", "red")

        if termination_reason:
            color = "green" if reward > 0 else "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)

        if mode == "Text":
            return trajectory
        if mode == "Token":
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                },
            }
            return token_result
        raise NotImplementedError(f"Unsupported mode: {mode}")

    async def execute_tasks(self, tasks: list[dict]):
        max_concurrent = self.n_parallel_agents

        all_trajectories = {}

        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initialized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    await index_queue.put(index)

        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]
        return ordered_trajectories


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
