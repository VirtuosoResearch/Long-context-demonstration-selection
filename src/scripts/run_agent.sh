export HF_TOKEN=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

sweagent run-batch \
  --config config/hf_llama_model.yaml \
  --agent.model.name huggingface/meta-llama/Llama-3.2-1B-Instruct:novita \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split dev \
  --instances.slice : \
  --instances.shuffle=True