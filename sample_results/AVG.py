import json
import os
from collections import defaultdict


def calculate_metric_averages(json_dir):
    """
    计算多个JSON文件中各指标的平均值

    参数:
        json_dir: JSON文件所在目录
    返回:
        按指标和类别分组的平均值字典
    """
    # 初始化存储结构: {类别: {指标: [值列表]}}
    metric_data = defaultdict(lambda: defaultdict(list))

    # 遍历目录下所有JSON文件
    for filename in os.listdir(json_dir):
        if filename.endswith("_final.json"):  # 匹配目标文件格式
            file_path = os.path.join(json_dir, filename)

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析文件 {filename}")
                    continue

            # 提取所有类别（overall + category_1至category_5）
            aggregate_metrics = data.get("aggregate_metrics", {})
            for category in aggregate_metrics:
                # 跳过非目标类别（如果有）
                if not (category == "overall" or category.startswith("category_")):
                    continue

                # 提取该类别下的所有指标
                metrics = aggregate_metrics[category]
                for metric_name, metric_info in metrics.items():
                    # 收集mean值
                    if "mean" in metric_info:
                        metric_data[category][metric_name].append(metric_info["mean"])

    # 计算平均值
    averages = defaultdict(dict)
    for category, metrics in metric_data.items():
        for metric_name, values in metrics.items():
            if values:  # 避免除以零
                averages[category][metric_name] = sum(values) / len(values)

    return averages


def print_averages(averages):
    """格式化输出平均值结果"""
    # 获取所有指标名称（用于对齐表头）
    all_metrics = set()
    for metrics in averages.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)

    # 打印表头
    print(f"{'类别':<12}" + "".join([f"{m:<18}" for m in all_metrics]))
    print("-" * (12 + 18 * len(all_metrics)))

    # 打印各分类数据
    for category in sorted(averages.keys()):
        row = [f"{category:<12}"]
        for metric in all_metrics:
            val = averages[category].get(metric, "-")
            row.append(f"{val:.4f}" if val != "-" else f"{val:<18}")
        print("        ".join(row))


# 使用示例
if __name__ == "__main__":
    # 替换为你的JSON文件目录
    json_directory = "."
    averages = calculate_metric_averages(json_directory)
    print_averages(averages)