import pandas as pd
from datasets import load_from_disk, DatasetDict, Dataset
import ast
import os

df = pd.read_csv('./notebooks/swebench_lite_classification.csv')
df['category'] = df['category'].apply(lambda x: x.strip())

instance2category = {}
for i, row in df.iterrows():
    cats = [c.strip() for c in row['category'].split(',')]
    instance2category[row['instance_id']] = cats

dataset_path = './datasets/oracle_lite_test/test'
dataset = load_from_disk(dataset_path)

categories = ['bugfix', 'feature', 'docs', 'refactor', 'compat', 'test']
category2indices = {cat: [] for cat in categories}

for idx, item in enumerate(dataset):
    instance_id = item['instance_id']
    cats = instance2category.get(instance_id, [])
    for cat in cats:
        if cat in category2indices:
            category2indices[cat].append(idx)

for cat in categories:
    sub_data = dataset.select(category2indices[cat])
    save_path = f'./datasets/oracle_lite_test_{cat}'
    os.makedirs(save_path, exist_ok=True)
    sub_data.save_to_disk(save_path)
    print(f'{cat} has saved to {save_path}')