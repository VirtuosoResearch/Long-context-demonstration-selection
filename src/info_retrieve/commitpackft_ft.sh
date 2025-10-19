python commitpackft_ft.py \
  --model_name codellama/CodeLlama-7b-Instruct-hf \
  --output_dir ./out/codellama-qlora-cpp \
  --cp_subset "c++" \
  --paper_recipe true --paper_model octocoder \
  --eval_steps 35 --logging_steps 5