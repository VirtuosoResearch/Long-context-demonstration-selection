# We write a simple function to wrapup the HuggingFace model loading and inference

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typer import prompt

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"    # swap if you prefer another instruct model
LOAD_8BIT    = False                           # set True if you installed bitsandbytes and want 8-bit loading
DTYPE        = torch.bfloat16 if torch.cuda.is_available() else torch.float32

class HF_LLM:
    
    def __init__(self, model_name=MODEL_NAME, load_8bit=LOAD_8BIT, dtype=DTYPE, max_new_tokens=160, generation_kwargs={}):
        self.model_name = model_name
        self.load_8bit = load_8bit
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = generation_kwargs

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # disable flash attention
            **({"load_in_8bit": True} if self.load_8bit else {})
        )
        
        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.generation_kwargs.get("temperature", 0.3),        # low-ish for format compliance
            top_p=self.generation_kwargs.get("top_p", 0.9),
            do_sample=self.generation_kwargs.get("do_sample", True)
        )
        self.stop_sequence = self.generation_kwargs.get("stop", "\n")
        self.format_guard = self.generation_kwargs.get(
            "format_guard", "" # empty by default
        )

    def __call__(self, prompt: str) -> str:
        """
        Completes from your existing ReAct prompt and returns exactly two lines:
        'Thought: ...' and 'Action: ...'
        """
        # We add a strong instruction to the prompt to improve compliance with the format
        full_prompt = prompt + self.format_guard

        #     Here, let's write the code to use language model to generate the response given the full_prompt
        #     First, we need to use the tokenizer to tokenize the prompt into pytorch tensors
        #     Second, we need to use model.generate() to generate the model response (which includes the Thought and Action)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=self.gen_cfg)

        # Slice off the prompt tokens to get only the completion
        completion = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Stop
        if self.stop_sequence in completion:
            completion = completion.split(self.stop_sequence)[0]

        return completion


