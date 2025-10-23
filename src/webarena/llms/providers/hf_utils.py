from text_generation import Client


def generate_from_huggingface_completion(
    prompt: str,
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    client = Client(model_endpoint, timeout=60)
    generation: str = client.generate(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
    ).generated_text

    return generation

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_from_local_model(
    prompt: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_new_tokens: int = 384,
    stop_sequences: list[str] | None = None,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("##############################################")
    print("INPUT_PROMPT",prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_text = output_text.split("assistant")[-1]
    print("##############################################")
    print("OUTPUT_TEXT: ",output_text)
    # exit(0)
    if stop_sequences:
        for stop in stop_sequences:
            if stop in output_text:
                output_text = output_text.split(stop)[0]

    return output_text