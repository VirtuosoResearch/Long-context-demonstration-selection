import asyncio
import concurrent.futures
import logging
import time
import openai
import torch
from openai.types import Completion

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

        self.agent_class = agent_class
        self.agent_args = agent_args
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
        async def get_response(prompt_text: str):
            retries = self.api_retries
            while retries > 0:
                try:
                    response = await self.client.completions.create(
                        prompt=prompt_text,
                        timeout=3600,
                        **self.sampling_params,
                        **kwargs,
                    )
                    return response
                except openai.RateLimitError:
                    retries -= 1
                    if retries == 0:
                        return "Error: Rate limit reached and retries exhausted."
                    logger.info("Sleep for 5 seconds for API limit.")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error("Error: %s", e)
                    return f"Error processing content: {e}"

        prompt_text = prompt
        if isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt):
            prompt_text = self.chat_parser.parse(prompt, add_generation_prompt=True, is_first_msg=True)

        response = await get_response(prompt_text)
        if isinstance(response, Completion):
            response = response.choices[0].text
        return response

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

            start_time = time.time()
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            action: Action = agent.update_from_model(response)
            action = action.action

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
