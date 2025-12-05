import os
import json
from collections import defaultdict

# 配置路径和文件列表
json_dir = "."  # 修改为你存放10个json文件的目录
output_file = "aggregated_metrics_mean.json"  # 最终输出文件

# 存储 overall 和各 category 的 mean 值
overall_stats = defaultdict(list)
category_stats = defaultdict(lambda: defaultdict(list))  # category_stats[metric_name][category_key]

# 遍历所有 .json 文件
for filename in sorted(os.listdir(json_dir)):
    if filename.endswith("final.json"):
        filepath = os.path.join(json_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取 overall 的所有 metrics 的 mean
        overall = data["aggregate_metrics"]["overall"]
        for metric, values in overall.items():
            if "mean" in values:
                overall_stats[metric].append(values["mean"])

        # 提取每个 category 的 metrics
        for key, cat_data in data["aggregate_metrics"].items():
            if key.startswith("category_") or key.isdigit():  # 支持 "category_1" 或 "1"
                for metric, values in cat_data.items():
                    if "mean" in values:
                        category_stats[metric][key].append(values["mean"])

# 计算 overall 每个 metric 的总平均
final_overall = {}
for metric, values in overall_stats.items():
    final_overall[metric] = sum(values) / len(values)

# 计算每个 category 下每个 metric 的总平均
final_category = {}
for metric, cat_dict in category_stats.items():
    final_category[metric] = {}
    for cat_key, values in cat_dict.items():
        final_category[metric][cat_key] = sum(values) / len(values)

# 整合成最终结构
final_result = {
    "model": "gpt-4o-mini",  # 示例保留 model 字段
    "dataset_count": len([f for f in os.listdir(json_dir) if f.endswith(".json")]),
    "aggregate_metrics": {
        "overall": {k: {"mean": v} for k, v in final_overall.items()}
    }
}

# 添加 category 结果
for metric, cats in final_category.items():
    for cat_key, mean_val in cats.items():
        if cat_key not in final_result["aggregate_metrics"]:
            final_result["aggregate_metrics"][cat_key] = {}
        final_result["aggregate_metrics"][cat_key][metric] = {"mean": mean_val}

# 保存到文件
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(final_result, out_f, indent=2, ensure_ascii=False)

print(f"✅ 已完成 10 个文件的聚合统计，结果保存至: {output_file}")
