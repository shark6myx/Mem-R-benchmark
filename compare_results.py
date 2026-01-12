"""
性能对比脚本

比较两个结果文件的性能指标，计算提升百分比
支持交互式文件选择
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def load_json(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_result_files(base_dir: str = ".") -> list:
    """查找所有结果文件"""
    result_files = []
    base_path = Path(base_dir)
    
    # 查找常见的结果目录
    search_dirs = [
        base_path / "base_sample_results",
        base_path / "sample_results",
        base_path / "reranking_sample_results",
        base_path
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for file in search_dir.glob("*_final.json"):
                result_files.append(str(file))
    
    return sorted(result_files)


def interactive_file_selection() -> tuple:
    """交互式选择文件"""
    files = find_result_files()
    
    if not files:
        print("未找到结果文件！")
        return None, None
    
    print("\n找到以下结果文件：")
    print("-" * 80)
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")
    print("-" * 80)
    
    # 选择基准文件
    while True:
        try:
            base_idx = input("\n请选择基准文件 (Base) 的编号: ").strip()
            base_idx = int(base_idx) - 1
            if 0 <= base_idx < len(files):
                base_file = files[base_idx]
                break
            else:
                print(f"请输入 1-{len(files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消")
            return None, None
    
    # 选择对比文件
    while True:
        try:
            compare_idx = input("请选择对比文件 (Compare) 的编号: ").strip()
            compare_idx = int(compare_idx) - 1
            if 0 <= compare_idx < len(files):
                compare_file = files[compare_idx]
                break
            else:
                print(f"请输入 1-{len(files)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消")
            return None, None
    
    return base_file, compare_file


def extract_metrics(data: Dict[str, Any], category: str = "overall") -> Dict[str, float]:
    """提取指定类别的指标"""
    metrics = {}
    if "aggregate_metrics" in data and category in data["aggregate_metrics"]:
        category_data = data["aggregate_metrics"][category]
        for metric_name, metric_data in category_data.items():
            if isinstance(metric_data, dict) and "mean" in metric_data:
                metrics[metric_name] = metric_data["mean"]
    return metrics


def calculate_improvement(old_value: float, new_value: float) -> tuple:
    """计算提升百分比和绝对提升"""
    if old_value == 0:
        if new_value == 0:
            return 0.0, 0.0, "无变化"
        else:
            return float('inf'), new_value, "提升"
    
    absolute_improvement = new_value - old_value
    percentage_improvement = (absolute_improvement / old_value) * 100
    
    if absolute_improvement > 0:
        direction = "提升"
    elif absolute_improvement < 0:
        direction = "下降"
    else:
        direction = "无变化"
    
    return percentage_improvement, absolute_improvement, direction


def format_number(value: float, decimals: int = 4) -> str:
    """格式化数字"""
    if value == float('inf'):
        return "∞"
    return f"{value:.{decimals}f}"


def compare_files(base_file: str, compare_file: str, show_all: bool = False):
    """比较两个结果文件"""
    print("\n" + "=" * 100)
    print("性能指标对比分析")
    print("=" * 100)
    print(f"\n基准文件 (Base): {base_file}")
    print(f"对比文件 (Compare): {compare_file}")
    print("=" * 100)
    
    # 加载文件
    try:
        base_data = load_json(base_file)
        compare_data = load_json(compare_file)
    except Exception as e:
        print(f"\n错误: 无法加载文件 - {e}")
        return
    
    # 提取所有类别的指标
    categories = ["overall", "category_1", "category_2", "category_3", "category_4", "category_5"]
    category_names = {
        "overall": "总体",
        "category_1": "类别1",
        "category_2": "类别2",
        "category_3": "类别3",
        "category_4": "类别4",
        "category_5": "类别5"
    }
    
    # 指标名称映射
    metric_names = {
        "exact_match": "精确匹配 (Exact Match)",
        "f1": "F1分数",
        "rouge1_f": "ROUGE-1 F",
        "rouge2_f": "ROUGE-2 F",
        "rougeL_f": "ROUGE-L F",
        "bleu1": "BLEU-1",
        "bleu2": "BLEU-2",
        "bleu3": "BLEU-3",
        "bleu4": "BLEU-4",
        "bert_precision": "BERT Precision",
        "bert_recall": "BERT Recall",
        "bert_f1": "BERT F1",
        "meteor": "METEOR",
        "sbert_similarity": "SBERT相似度"
    }
    
    # 为每个类别生成对比表
    for category in categories:
        base_metrics = extract_metrics(base_data, category)
        compare_metrics = extract_metrics(compare_data, category)
        
        if not base_metrics or not compare_metrics:
            continue
        
        print(f"\n{'=' * 100}")
        print(f"{category_names[category]} ({category.upper()})")
        print(f"{'=' * 100}")
        
        # 表头
        print(f"\n{'指标':<30} {'基准值 (Base)':<18} {'对比值 (Compare)':<18} {'绝对提升':<15} {'提升百分比':<15} {'状态':<10}")
        print("-" * 100)
        
        # 获取所有指标
        all_metrics = set(base_metrics.keys()) | set(compare_metrics.keys())
        sorted_metrics = sorted(all_metrics)
        
        improvements = []
        
        for metric in sorted_metrics:
            base_val = base_metrics.get(metric, 0.0)
            compare_val = compare_metrics.get(metric, 0.0)
            
            # 注意：这里计算的是 base 相比 compare 的提升
            pct_improvement, abs_improvement, direction = calculate_improvement(compare_val, base_val)
            
            metric_display = metric_names.get(metric, metric)
            
            # 格式化输出
            base_str = format_number(base_val, 4)
            compare_str = format_number(compare_val, 4)
            abs_str = format_number(abs_improvement, 4)
            
            if pct_improvement == float('inf'):
                pct_str = "∞"
            else:
                pct_str = f"{pct_improvement:+.2f}%"
            
            # 根据方向设置状态符号
            if direction == "提升":
                status = "↑ 提升"
            elif direction == "下降":
                status = "↓ 下降"
            else:
                status = "→ 无变化"
            
            print(f"{metric_display:<30} {base_str:<18} {compare_str:<18} {abs_str:<15} {pct_str:<15} {status:<10}")
            
            improvements.append({
                "metric": metric,
                "base": base_val,
                "compare": compare_val,
                "improvement": abs_improvement,
                "percentage": pct_improvement,
                "direction": direction
            })
        
        # 统计摘要
        print("\n" + "-" * 100)
        total_improvements = sum(1 for imp in improvements if imp["direction"] == "提升")
        total_decreases = sum(1 for imp in improvements if imp["direction"] == "下降")
        total_no_change = sum(1 for imp in improvements if imp["direction"] == "无变化")
        
        if total_improvements > 0:
            avg_improvement = sum(imp["improvement"] for imp in improvements if imp["direction"] == "提升") / total_improvements
            print(f"提升的指标数: {total_improvements} (平均提升: {avg_improvement:.4f})")
        if total_decreases > 0:
            avg_decrease = sum(imp["improvement"] for imp in improvements if imp["direction"] == "下降") / total_decreases
            print(f"下降的指标数: {total_decreases} (平均下降: {abs(avg_decrease):.4f})")
        if total_no_change > 0:
            print(f"无变化的指标数: {total_no_change}")
    
    # 总体总结
    print(f"\n{'=' * 100}")
    print("关键指标总结")
    print(f"{'=' * 100}")
    
    overall_base = extract_metrics(base_data, "overall")
    overall_compare = extract_metrics(compare_data, "overall")
    
    key_metrics = ["exact_match", "f1", "rouge1_f", "rougeL_f", "bert_f1", "meteor"]
    
    print(f"\n{'指标':<30} {'基准值':<15} {'对比值':<15} {'提升':<15} {'提升%':<15}")
    print("-" * 90)
    
    for metric in key_metrics:
        if metric in overall_base and metric in overall_compare:
            base_val = overall_base[metric]
            compare_val = overall_compare[metric]
            pct_improvement, abs_improvement, direction = calculate_improvement(compare_val, base_val)
            
            metric_display = metric_names.get(metric, metric)
            base_str = format_number(base_val, 4)
            compare_str = format_number(compare_val, 4)
            abs_str = format_number(abs_improvement, 4)
            
            if pct_improvement == float('inf'):
                pct_str = "∞"
            else:
                pct_str = f"{pct_improvement:+.2f}%"
            
            print(f"{metric_display:<30} {base_str:<15} {compare_str:<15} {abs_str:<15} {pct_str:<15}")


def main():
    parser = argparse.ArgumentParser(description="比较两个结果文件的性能指标")
    parser.add_argument("--base", type=str, default=None,
                       help="基准文件路径（如果不指定，将交互式选择）")
    parser.add_argument("--compare", type=str, default=None,
                       help="对比文件路径（如果不指定，将交互式选择）")
    parser.add_argument("--all", action="store_true",
                       help="显示所有指标（包括BERT等）")
    
    args = parser.parse_args()
    
    # 如果两个文件都指定了，直接使用
    if args.base and args.compare:
        base_file = args.base
        compare_file = args.compare
    else:
        # 交互式选择
        base_file, compare_file = interactive_file_selection()
        if not base_file or not compare_file:
            return
    
    compare_files(base_file, compare_file, show_all=args.all)


if __name__ == "__main__":
    main()

