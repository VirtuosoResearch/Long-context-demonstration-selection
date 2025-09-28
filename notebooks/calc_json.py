import os
import json
from collections import defaultdict

base_path = "./logs/run_evaluation/test_gold_all/gold"
classified_json = "./classified/SWE-bench_bm25_13K_test.json"

with open(classified_json, "r") as f:
    categories = json.load(f)

stats = defaultdict(lambda: {"total": 0, "with_report": 0})

for category, instances in categories.items():
    for inst in instances:
        inst_path = os.path.join(base_path, inst)
        if not os.path.exists(inst_path): continue
        stats[category]["total"] += 1
        report_file = os.path.join(inst_path, "report.json")
        if os.path.exists(report_file):
            stats[category]["with_report"] += 1

sum_total , sum_wr= 0,0
for category, res in stats.items():
    print(f"{category}: total={res['total']}, with_report={res['with_report']}, rate={res['with_report']/res['total']:.3f}")
    sum_total+=res["total"]; sum_wr+=res["with_report"]
print(sum_total, sum_wr)