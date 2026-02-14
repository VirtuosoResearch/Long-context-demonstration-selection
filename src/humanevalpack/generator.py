import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

class Generator:
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temp: float = 0.2,
        top_p: float = 0.95,
        history: str = "",
        latest_error: str = "",
    ) -> str:
        raise NotImplementedError


class TransformersGenerator(Generator):
    def __init__(self, model_name: str, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.is_chat = getattr(self.tokenizer, "chat_template", None) not in (None, "", False)

    def _make_inputs(self, task_prompt: str, lang: str, history: str = "", latest_error: str = "") -> str:
        # Language-specific system/user instructions
        if lang == "js":
            sys_inst = (
                "You are a meticulous JavaScript (Node.js) engineer.\n"
                "Complete the requested function so that it passes the provided tests.\n"
                "Return ONLY valid JavaScript code (no markdown fences, no explanations)."
            )
            user_inst = (
                "Write modern JS (ES6+), no TypeScript, no external libraries. "
                "Avoid printing debug logs. Provide only function(s) and helpers."
            )
        elif lang == "go":
            sys_inst = (
                "You are a meticulous Go engineer.\n"
                "Complete the requested function so that it passes the provided tests.\n"
                "Return ONLY valid Go code (no markdown fences, no explanations)."
            )
            user_inst = (
                "Write compatible Go 1.18+ code. Do not use external modules. "
                "If the prompt declares a package or function signature, implement accordingly."
            )
        elif lang == "rust":
            sys_inst = (
                "You are a meticulous Rust engineer.\n"
                "Complete the requested function so that it passes the provided tests.\n"
                "Return ONLY valid Rust code (no markdown fences, no explanations)."
            )
            user_inst = (
                "Write stable Rust (edition 2021). No external crates. "
                "Provide function(s) and any helper functions required."
            )
        elif lang == "python":
            sys_inst = (
                "You are a meticulous software engineer. "
                "Complete the Python function below so that it passes the provided tests. "
                "Return ONLY valid Python code. Do not use markdown fences."
            )
            user_inst = (
                "Fill in the function by writing correct, efficient Python. "
                "Do not include explanations—only executable code that defines the function."
            )
        elif lang == "java":
            sys_inst = (
                "You are a meticulous Java software engineer.\n"
                "Complete the requested Java method/class so that it passes the provided tests.\n"
                "Return ONLY valid Java code (no markdown fences, no explanations)."
            )
            user_inst = (
                "Write correct, efficient Java 8+ (no external libs). "
                "If the prompt declares a class, implement inside it; otherwise provide method implementation (and helpers) only. "
                "Do NOT print debug logs."
            )
        elif lang == "cpp":
            sys_inst = (
                "You are a meticulous C++ software engineer.\n"
                "Complete the requested C++ function so that it passes the provided tests.\n"
                "Return ONLY valid C++ code for the function body (and any necessary helper functions), "
                "without markdown fences or explanations."
            )
            user_inst = (
                "Write correct, efficient, standard C++17 code. Do not print debug info. "
                "Do not include a main() unless explicitly asked in the prompt."
            )
        else:
            raise ValueError("Unsupported lang")

        history_text = history.strip() if history else ""
        error_text = latest_error.strip() if latest_error else ""
        if history_text:
            user_inst = (
                user_inst
                + "\n\n"
            )
            prompt_parts = [
                f"### ORIGINAL TASK\n{task_prompt}",
                f"You previously produced an incorrect solution for this same task. "
                f"Read the full attempt history and fix the bug in the next answer. "
                f"Return only corrected code.",
                f"### ATTEMPT HISTORY (STEP 1 TO PREVIOUS STEP)\n{history_text}",
            ]
            history_lower = history_text.lower()
            has_feedback = ("feedback:" in history_lower) or ("compiler output" in history_lower) or ("runtime output" in history_lower)
            if error_text and not has_feedback:
                prompt_parts.append(f"### LATEST FEEDBACK\n{error_text}")
            prompt_parts.append("### FIX THE BUG\nOutput only corrected code.")
            final_prompt = "\n\n".join(prompt_parts)
        else:
            final_prompt = task_prompt

        if self.is_chat:
            messages = [
                {"role": "system", "content": sys_inst},
                {"role": "user", "content": f"{user_inst}\n\n{final_prompt}"},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return sys_inst + "\n\n" + user_inst + "\n\n" + final_prompt

    @staticmethod
    def _strip_fences(s: str) -> str:
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        return s.strip()

    def _tail_from_generated(self, inputs_ids, full_text: str) -> str:
        prompt_text = self.tokenizer.decode(inputs_ids["input_ids"][0], skip_special_tokens=True)
        return full_text[len(prompt_text):]

    def generate(
        self,
        prompt: str,
        lang: str,
        max_new_tokens: int = 256,
        temp: float = 0.2,
        top_p: float = 0.95,
        history: str = "",
        latest_error: str = "",
    ) -> str:
        import torch
        inputs = self._make_inputs(prompt, lang, history=history, latest_error=latest_error)
        input_ids = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **input_ids,
                do_sample=True if temp > 0 else False,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        tail = self._tail_from_generated(input_ids, text)
        return self._strip_fences(tail)
