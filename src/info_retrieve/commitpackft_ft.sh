python commitpackft_ft.py \
  --model_name codellama/CodeLlama-7b-Instruct-hf \
  --output_dir ./out/codellama-qlora-python_100 \
  --cp_subset "python" \
  --paper_recipe true --paper_model octocoder \
  --eval_steps 100 --logging_steps 5