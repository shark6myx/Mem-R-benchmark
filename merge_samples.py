import json
import os
from collections import defaultdict

# 请修改为你的结果目录（即上面代码中的 sample_results/eval_xxx 文件夹）
RESULT_DIR = "./sample_results/eval_gpt-4o-mini_2025-11-12-10-00"  # 替换为实际路径
OUTPUT_FILE = "final_merged_results.json"


def merge():
    all_results = []
    # 遍历所有sample结果文件
    for filename in os.listdir(RESULT_DIR):
        if filename.startswith("result_sample_") and filename.endswith(".json"):
            with open(os.path.join(RESULT_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.extend(data["individual_results"])

    # 生成合并报告
    merged = {
        "total_samples": len(set(r["sample_id"] for r in all_results)),
        "total_questions": len(all_results),
        "all_results": all_results
    }

    # 保存合并结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"合并完成！共{len(all_results)}个问题，结果保存在：{OUTPUT_FILE}")


if __name__ == "__main__":
    merge()