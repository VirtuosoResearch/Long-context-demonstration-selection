# python eval_qa.py --model_name Qwen/Qwen3-1.7B --n_eval 500 --device 0 --task_name fever

python eval_qa.py --model_name meta-llama/Llama-3.2-3B --n_eval 500 --device 0
python eval_qa.py --model_name Qwen/Qwen3-4B --n_eval 500 --device 0
python eval_qa.py --model_name Qwen/Qwen3-8B --n_eval 500 --device 0
python eval_qa.py --model_name deepseek-ai/deepseek-llm-7b-chat --n_eval 500 --device 0

python eval_qa.py --model_name meta-llama/Llama-3.2-3B --n_eval 500 --device 0 --task_name fever    
python eval_qa.py --model_name Qwen/Qwen3-4B --n_eval 500 --device 0 --task_name fever
python eval_qa.py --model_name Qwen/Qwen3-8B --n_eval 500 --device 0 --task_name fever
python eval_qa.py --model_name deepseek-ai/deepseek-llm-7b-chat --n_eval 500 --device 0 --task_name fever
