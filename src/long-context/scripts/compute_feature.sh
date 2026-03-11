# python get_feature_causal.py --task cr --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 8 --max_length 128
python get_feature_causal.py --task sst2 --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 8 --max_length 128
python get_feature_causal.py --task poem_sentiment --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 8 --max_length 128
python get_feature_causal.py --tasks glue-rte --model_name Qwen/Qwen2.5-1.5B-Instruct --batch_size 8 --max_length 128