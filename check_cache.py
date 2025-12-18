"""
缓存文件检测脚本

用于检测和验证 runsingle.py 使用的缓存文件是否能正确读取
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from memory_layer import HybridRetriever, MemoryNote


def find_cache_directories(base_dir: str = ".") -> List[str]:
    """
    查找所有缓存目录
    
    参数:
        base_dir: 基础目录路径
        
    返回:
        缓存目录路径列表
    """
    cache_dirs = []
    base_path = Path(base_dir)
    
    # 查找所有符合命名模式的目录
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("cached_memories_advanced_"):
            cache_dirs.append(str(item))
    
    return sorted(cache_dirs)


def check_cache_directory(cache_dir: str, sample_idx: Optional[int] = None) -> Dict:
    """
    检查单个缓存目录
    
    参数:
        cache_dir: 缓存目录路径
        sample_idx: 要检查的样本索引（如果为None，检查所有样本）
        
    返回:
        检查结果字典
    """
    cache_path = Path(cache_dir)
    result = {
        "cache_dir": cache_dir,
        "exists": cache_path.exists(),
        "files": [],
        "samples": {},
        "errors": []
    }
    
    if not cache_path.exists():
        result["errors"].append(f"缓存目录不存在: {cache_dir}")
        return result
    
    # 列出所有文件
    all_files = list(cache_path.iterdir())
    result["files"] = [f.name for f in all_files if f.is_file()]
    
    # 解析目录名获取backend和model
    dir_name = cache_path.name
    parts = dir_name.replace("cached_memories_advanced_", "").split("_", 1)
    if len(parts) == 2:
        backend, model = parts
    else:
        backend, model = "unknown", "unknown"
    
    result["backend"] = backend
    result["model"] = model
    
    # 找出所有样本索引
    memory_files = [f for f in all_files if f.name.startswith("memory_cache_sample_") and f.suffix == ".pkl"]
    sample_indices = set()
    
    for mem_file in memory_files:
        try:
            # 提取样本索引：memory_cache_sample_9.pkl -> 9
            idx_str = mem_file.stem.replace("memory_cache_sample_", "")
            sample_indices.add(int(idx_str))
        except ValueError:
            pass
    
    # 如果指定了sample_idx，只检查该样本
    if sample_idx is not None:
        sample_indices = {sample_idx} if sample_idx in sample_indices else set()
    
    # 检查每个样本的缓存文件
    for idx in sorted(sample_indices):
        sample_result = check_sample_cache(cache_path, idx)
        result["samples"][idx] = sample_result
    
    return result


def check_sample_cache(cache_dir: Path, sample_idx: int) -> Dict:
    """
    检查单个样本的缓存文件
    
    参数:
        cache_dir: 缓存目录路径
        sample_idx: 样本索引
        
    返回:
        样本缓存检查结果
    """
    result = {
        "sample_idx": sample_idx,
        "memory_cache": {"exists": False, "readable": False, "count": 0, "error": None},
        "retriever_cache": {"exists": False, "readable": False, "corpus_size": 0, "error": None},
        "embeddings_cache": {"exists": False, "readable": False, "shape": None, "error": None}
    }
    
    # 检查记忆缓存文件
    memory_file = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    result["memory_cache"]["exists"] = memory_file.exists()
    
    if memory_file.exists():
        try:
            with open(memory_file, 'rb') as f:
                cached_memories = pickle.load(f)
            
            if isinstance(cached_memories, dict):
                result["memory_cache"]["readable"] = True
                result["memory_cache"]["count"] = len(cached_memories)
                # 验证是否是MemoryNote对象
                if cached_memories:
                    first_key = list(cached_memories.keys())[0]
                    first_value = cached_memories[first_key]
                    result["memory_cache"]["type"] = type(first_value).__name__
                    if hasattr(first_value, 'content'):
                        result["memory_cache"]["sample_content"] = first_value.content[:100] + "..." if len(first_value.content) > 100 else first_value.content
            else:
                result["memory_cache"]["error"] = f"缓存格式错误：期望字典，得到 {type(cached_memories).__name__}"
        except Exception as e:
            result["memory_cache"]["error"] = str(e)
    
    # 检查检索器缓存文件
    retriever_file = cache_dir / f"retriever_cache_sample_{sample_idx}.pkl"
    result["retriever_cache"]["exists"] = retriever_file.exists()
    
    if retriever_file.exists():
        try:
            with open(retriever_file, 'rb') as f:
                state = pickle.load(f)
            
            if isinstance(state, dict):
                result["retriever_cache"]["readable"] = True
                result["retriever_cache"]["corpus_size"] = len(state.get("corpus", []))
                result["retriever_cache"]["has_bm25"] = "bm25" in state
                result["retriever_cache"]["has_document_ids"] = "document_ids" in state
                result["retriever_cache"]["alpha"] = state.get("alpha", "N/A")
                result["retriever_cache"]["model_name"] = state.get("model_name", "N/A")
                
                # 尝试加载检索器
                embeddings_file = cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"
                if embeddings_file.exists():
                    try:
                        retriever = HybridRetriever.load(str(retriever_file), str(embeddings_file))
                        result["retriever_cache"]["loadable"] = True
                        result["retriever_cache"]["retriever_type"] = type(retriever).__name__
                    except Exception as e:
                        result["retriever_cache"]["load_error"] = str(e)
        except Exception as e:
            result["retriever_cache"]["error"] = str(e)
    
    # 检查嵌入向量缓存文件
    embeddings_file = cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    result["embeddings_cache"]["exists"] = embeddings_file.exists()
    
    if embeddings_file.exists():
        try:
            embeddings = np.load(embeddings_file)
            result["embeddings_cache"]["readable"] = True
            result["embeddings_cache"]["shape"] = embeddings.shape
            result["embeddings_cache"]["dtype"] = str(embeddings.dtype)
        except Exception as e:
            result["embeddings_cache"]["error"] = str(e)
    
    return result


def print_check_results(results: Dict):
    """
    打印检查结果
    
    参数:
        results: check_cache_directory返回的结果字典
    """
    print("=" * 80)
    print(f"缓存目录: {results['cache_dir']}")
    print(f"目录存在: {'✓' if results['exists'] else '✗'}")
    
    if not results['exists']:
        for error in results['errors']:
            print(f"  错误: {error}")
        return
    
    print(f"Backend: {results.get('backend', 'unknown')}")
    print(f"Model: {results.get('model', 'unknown')}")
    print(f"文件总数: {len(results['files'])}")
    
    if results['files']:
        print(f"\n文件列表（前10个）:")
        for fname in sorted(results['files'])[:10]:
            file_path = Path(results['cache_dir']) / fname
            size = file_path.stat().st_size / 1024  # KB
            print(f"  - {fname} ({size:.2f} KB)")
        if len(results['files']) > 10:
            print(f"  ... 还有 {len(results['files']) - 10} 个文件")
    
    # 打印每个样本的检查结果
    if results['samples']:
        print(f"\n样本检查结果:")
        for sample_idx in sorted(results['samples'].keys()):
            sample_result = results['samples'][sample_idx]
            print(f"\n  样本 {sample_idx}:")
            
            # 记忆缓存
            mem_cache = sample_result['memory_cache']
            status = "✓" if mem_cache['readable'] else ("?" if mem_cache['exists'] else "✗")
            print(f"    记忆缓存: {status}")
            if mem_cache['exists']:
                if mem_cache['readable']:
                    print(f"      - 记忆数量: {mem_cache['count']}")
                    print(f"      - 类型: {mem_cache.get('type', 'N/A')}")
                    if 'sample_content' in mem_cache:
                        print(f"      - 示例内容: {mem_cache['sample_content']}")
                if mem_cache['error']:
                    print(f"      - 错误: {mem_cache['error']}")
            
            # 检索器缓存
            ret_cache = sample_result['retriever_cache']
            status = "✓" if ret_cache['readable'] else ("?" if ret_cache['exists'] else "✗")
            print(f"    检索器缓存: {status}")
            if ret_cache['exists']:
                if ret_cache['readable']:
                    print(f"      - 语料库大小: {ret_cache['corpus_size']}")
                    print(f"      - Alpha值: {ret_cache.get('alpha', 'N/A')}")
                    print(f"      - 模型名称: {ret_cache.get('model_name', 'N/A')}")
                    print(f"      - 包含BM25: {ret_cache.get('has_bm25', False)}")
                    if 'loadable' in ret_cache:
                        print(f"      - 可加载: {'✓' if ret_cache['loadable'] else '✗'}")
                    if 'load_error' in ret_cache:
                        print(f"      - 加载错误: {ret_cache['load_error']}")
                if ret_cache['error']:
                    print(f"      - 错误: {ret_cache['error']}")
            
            # 嵌入向量缓存
            emb_cache = sample_result['embeddings_cache']
            status = "✓" if emb_cache['readable'] else ("?" if emb_cache['exists'] else "✗")
            print(f"    嵌入向量缓存: {status}")
            if emb_cache['exists']:
                if emb_cache['readable']:
                    print(f"      - 形状: {emb_cache['shape']}")
                    print(f"      - 数据类型: {emb_cache.get('dtype', 'N/A')}")
                if emb_cache['error']:
                    print(f"      - 错误: {emb_cache['error']}")
            
            # 总体状态
            all_good = (
                mem_cache['readable'] and
                ret_cache['readable'] and
                emb_cache['readable']
            )
            print(f"    总体状态: {'✓ 完整' if all_good else '⚠ 不完整'}")


def simulate_runsingle_cache_load(cache_dir: str, backend: str, model: str, sample_idx: int):
    """
    模拟 runsingle.py 的缓存加载过程
    
    参数:
        cache_dir: 缓存目录路径
        backend: backend参数
        model: model参数
        sample_idx: 样本索引
    """
    print("\n" + "=" * 80)
    print("模拟 runsingle.py 的缓存加载过程")
    print("=" * 80)
    
    base_dir = Path(".")
    expected_cache_dir = base_dir / f"cached_memories_advanced_{backend}_{model}"
    
    print(f"\n预期缓存目录: {expected_cache_dir}")
    print(f"目录存在: {'✓' if expected_cache_dir.exists() else '✗'}")
    
    if not expected_cache_dir.exists():
        print(f"⚠ 警告: 缓存目录不存在！")
        print(f"   当前backend参数: {backend}")
        print(f"   当前model参数: {model}")
        print(f"   请检查参数是否与缓存目录名匹配")
        return
    
    # 构建文件路径（与runsingle.py一致）
    memory_cache_file = expected_cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    retriever_cache_file = expected_cache_dir / f"retriever_cache_sample_{sample_idx}.pkl"
    retriever_cache_embeddings_file = expected_cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    
    print(f"\n查找的缓存文件:")
    print(f"  记忆缓存: {memory_cache_file}")
    print(f"    存在: {'✓' if memory_cache_file.exists() else '✗'}")
    print(f"  检索器缓存: {retriever_cache_file}")
    print(f"    存在: {'✓' if retriever_cache_file.exists() else '✗'}")
    print(f"  嵌入向量缓存: {retriever_cache_embeddings_file}")
    print(f"    存在: {'✓' if retriever_cache_embeddings_file.exists() else '✗'}")
    
    # 尝试加载（与runsingle.py逻辑一致）
    if memory_cache_file.exists():
        print(f"\n✓ 找到记忆缓存，可以加载")
        try:
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            print(f"  - 成功加载，记忆数量: {len(cached_memories)}")
            
            if retriever_cache_file.exists():
                print(f"\n✓ 找到检索器缓存，尝试加载...")
                try:
                    from memory_layer import HybridRetriever
                    retriever = HybridRetriever.load(str(retriever_cache_file), str(retriever_cache_embeddings_file))
                    print(f"  - 成功加载检索器")
                    print(f"  - 语料库大小: {len(retriever.corpus)}")
                    print(f"  - Alpha值: {retriever.alpha}")
                except Exception as e:
                    print(f"  - ✗ 加载失败: {e}")
                    print(f"  - 将使用 load_from_local_memory 重建检索器")
            else:
                print(f"\n⚠ 未找到检索器缓存，将使用 load_from_local_memory 重建")
        except Exception as e:
            print(f"\n✗ 加载记忆缓存失败: {e}")
    else:
        print(f"\n✗ 未找到记忆缓存，将创建新的记忆")


def main():
    parser = argparse.ArgumentParser(description="检测缓存文件")
    parser.add_argument("--dir", type=str, default=".", help="基础目录路径")
    parser.add_argument("--cache-dir", type=str, default=None, help="指定要检查的缓存目录（完整路径）")
    parser.add_argument("--sample", type=int, default=None, help="指定要检查的样本索引")
    parser.add_argument("--simulate", action="store_true", help="模拟runsingle.py的加载过程")
    parser.add_argument("--backend", type=str, default="sglang", help="backend参数（用于simulate）")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="model参数（用于simulate）")
    
    args = parser.parse_args()
    
    print("缓存文件检测工具")
    print("=" * 80)
    
    if args.simulate:
        # 模拟runsingle.py的加载过程
        if args.sample is None:
            print("错误: 使用 --simulate 时必须指定 --sample")
            return
        
        simulate_runsingle_cache_load(
            args.dir,
            args.backend,
            args.model,
            args.sample
        )
        return
    
    # 查找或使用指定的缓存目录
    if args.cache_dir:
        cache_dirs = [args.cache_dir]
    else:
        cache_dirs = find_cache_directories(args.dir)
    
    if not cache_dirs:
        print(f"在 {args.dir} 中未找到任何缓存目录")
        print(f"预期目录命名格式: cached_memories_advanced_<backend>_<model>")
        return
    
    print(f"找到 {len(cache_dirs)} 个缓存目录:\n")
    
    # 检查每个缓存目录
    for cache_dir in cache_dirs:
        results = check_cache_directory(cache_dir, args.sample)
        print_check_results(results)
        print()
    
    # 如果指定了sample，也进行模拟
    if args.sample is not None:
        print("\n提示: 使用 --simulate 参数可以模拟 runsingle.py 的加载过程")


if __name__ == "__main__":
    main()

