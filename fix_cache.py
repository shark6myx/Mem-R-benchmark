"""
缓存文件修复脚本

用于修复损坏的pickle缓存文件，特别是包含BOM或额外字节的文件
"""

import pickle
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import argparse


def check_file_header(file_path: Path) -> bytes:
    """检查文件的前几个字节"""
    with open(file_path, 'rb') as f:
        header = f.read(10)
    return header


def fix_pickle_file(file_path: Path, backup: bool = True) -> Tuple[bool, Optional[str]]:
    """
    尝试修复损坏的pickle文件
    
    参数:
        file_path: 文件路径
        backup: 是否备份原文件
        
    返回:
        (是否成功, 错误信息)
    """
    if not file_path.exists():
        return False, "文件不存在"
    
    print(f"\n处理文件: {file_path.name}")
    
    # 读取原始文件
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
    except Exception as e:
        return False, f"读取文件失败: {e}"
    
    file_size = len(raw_data)
    print(f"  文件大小: {file_size} 字节")
    
    # 检查文件开头
    header = raw_data[:min(10, file_size)]
    print(f"  文件前10个字节: {header.hex()} ({header})")
    
    # 备份原文件
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        try:
            shutil.copy2(file_path, backup_path)
            print(f"  [OK] 已备份原文件到: {backup_path.name}")
        except Exception as e:
            print(f"  [WARN] 备份失败: {e}")
    
    # 尝试不同的修复策略
    strategies = [
        ("原始数据", raw_data),
        ("跳过UTF-8 BOM (3字节)", raw_data[3:] if len(raw_data) > 3 else raw_data),
        ("跳过1字节", raw_data[1:] if len(raw_data) > 1 else raw_data),
        ("跳过2字节", raw_data[2:] if len(raw_data) > 2 else raw_data),
        ("跳过4字节", raw_data[4:] if len(raw_data) > 4 else raw_data),
        ("跳过6字节", raw_data[6:] if len(raw_data) > 6 else raw_data),
        ("跳过8字节", raw_data[8:] if len(raw_data) > 8 else raw_data),
    ]
    
    # 先尝试固定偏移策略
    for strategy_name, test_data in strategies:
        if len(test_data) == 0:
            continue
            
        try:
            # 尝试加载
            test_obj = pickle.loads(test_data)
            print(f"  [OK] 成功！使用策略: {strategy_name}")
            
            # 如果原始数据就能加载，说明文件没问题
            if strategy_name == "原始数据":
                print(f"  [INFO] 文件可以正常加载，无需修复")
                return True, None
            
            # 保存修复后的文件
            try:
                with open(file_path, 'wb') as f:
                    f.write(test_data)
                print(f"  [OK] 已保存修复后的文件")
                return True, None
            except Exception as e:
                return False, f"保存修复后的文件失败: {e}"
                
        except Exception as e:
            # 继续尝试下一个策略
            continue
    
    # 如果固定偏移都失败，尝试扫描文件找到有效的pickle起始位置
    print(f"  尝试扫描文件查找有效的pickle数据...")
    
    # Pickle文件通常以特定字节开始，尝试查找常见的pickle协议标记
    # Protocol 0-4的常见起始字节: 0x80, 0x81, 0x82, 0x83, 0x84
    # 或者直接查找字典开始的标记: 0x7d (EMPTY_DICT), 0x7b (EMPTY_DICT)
    
    pickle_markers = [b'\x80', b'\x81', b'\x82', b'\x83', b'\x84', b'\x7d', b'\x7b']
    
    for offset in range(min(100, len(raw_data))):  # 只扫描前100字节
        test_data = raw_data[offset:]
        if len(test_data) < 10:  # 太短的数据不太可能是有效的pickle
            continue
            
        # 检查是否以pickle标记开始
        if test_data[0:1] in pickle_markers:
            try:
                test_obj = pickle.loads(test_data)
                print(f"  [OK] 成功！在偏移 {offset} 字节处找到有效的pickle数据")
                # 保存修复后的文件
                try:
                    with open(file_path, 'wb') as f:
                        f.write(test_data)
                    print(f"  [OK] 已保存修复后的文件")
                    return True, None
                except Exception as e:
                    return False, f"保存修复后的文件失败: {e}"
            except:
                continue
    
    return False, "所有修复策略都失败了，文件可能严重损坏，建议删除并重新生成"


def fix_cache_directory(cache_dir: Path, sample_indices: Optional[list] = None, 
                        fix_memory: bool = True, fix_retriever: bool = True,
                        delete_corrupted: bool = False) -> dict:
    """
    修复缓存目录中的文件
    
    参数:
        cache_dir: 缓存目录路径
        sample_indices: 要修复的样本索引列表（None表示修复所有）
        fix_memory: 是否修复记忆缓存文件
        fix_retriever: 是否修复检索器缓存文件
        
    返回:
        修复结果统计
    """
    if not cache_dir.exists():
        print(f"错误: 缓存目录不存在: {cache_dir}")
        return {"success": 0, "failed": 0, "skipped": 0}
    
    print(f"缓存目录: {cache_dir}")
    print("=" * 80)
    
    # 找出所有需要修复的文件
    files_to_fix = []
    
    if sample_indices is None:
        # 找出所有样本索引
        memory_files = list(cache_dir.glob("memory_cache_sample_*.pkl"))
        sample_indices = set()
        for mem_file in memory_files:
            try:
                idx_str = mem_file.stem.replace("memory_cache_sample_", "")
                sample_indices.add(int(idx_str))
            except ValueError:
                pass
        sample_indices = sorted(sample_indices)
    
    print(f"找到 {len(sample_indices)} 个样本需要检查")
    
    for sample_idx in sample_indices:
        if fix_memory:
            memory_file = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
            if memory_file.exists():
                files_to_fix.append(("memory", memory_file, sample_idx))
        
        if fix_retriever:
            retriever_file = cache_dir / f"retriever_cache_sample_{sample_idx}.pkl"
            if retriever_file.exists():
                files_to_fix.append(("retriever", retriever_file, sample_idx))
    
    print(f"\n需要检查 {len(files_to_fix)} 个文件\n")
    
    results = {"success": 0, "failed": 0, "skipped": 0}
    
    for file_type, file_path, sample_idx in files_to_fix:
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx} - {file_type.upper()} 缓存")
        print(f"{'='*80}")
        
        # 先尝试正常加载，如果成功就跳过
        try:
            with open(file_path, 'rb') as f:
                pickle.load(f)
            print(f"  [OK] 文件可以正常加载，跳过修复")
            results["skipped"] += 1
            continue
        except Exception as e:
            print(f"  [ERROR] 文件加载失败: {e}")
            print(f"  开始尝试修复...")
        
        # 尝试修复
        success, error = fix_pickle_file(file_path, backup=True)
        
        if success:
            results["success"] += 1
            print(f"  [OK] 修复成功！")
        else:
            results["failed"] += 1
            print(f"  [FAIL] 修复失败: {error}")
            # 如果启用删除选项，删除损坏的文件
            if delete_corrupted:
                try:
                    file_path.unlink()
                    print(f"  [INFO] 已删除损坏的文件: {file_path.name}")
                except Exception as e:
                    print(f"  [WARN] 删除文件失败: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="修复损坏的缓存文件")
    parser.add_argument("--cache-dir", type=str, 
                       default="cached_memories_advanced_sglang_gpt-4o-mini",
                       help="缓存目录路径")
    parser.add_argument("--sample", type=int, nargs="+", default=None,
                       help="指定要修复的样本索引（例如: --sample 8 9）")
    parser.add_argument("--no-memory", action="store_true",
                       help="不修复记忆缓存文件")
    parser.add_argument("--no-retriever", action="store_true",
                       help="不修复检索器缓存文件")
    parser.add_argument("--check-only", action="store_true",
                       help="仅检查文件，不进行修复")
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    print("=" * 80)
    print("缓存文件修复工具")
    print("=" * 80)
    
    if args.check_only:
        print("\n[仅检查模式]")
        # 只检查，不修复
        for sample_idx in (args.sample or [8, 9]):
            memory_file = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
            if memory_file.exists():
                print(f"\n检查 Sample {sample_idx}:")
                header = check_file_header(memory_file)
                print(f"  文件头: {header.hex()}")
                try:
                    with open(memory_file, 'rb') as f:
                        pickle.load(f)
                    print(f"  [OK] 文件可以正常加载")
                except Exception as e:
                    print(f"  [ERROR] 文件损坏: {e}")
        return
    
    # 执行修复（传递delete_corrupted参数）
    args_dict = vars(args)
    results = fix_cache_directory(
        cache_dir,
        sample_indices=args.sample,
        fix_memory=not args.no_memory,
        fix_retriever=not args.no_retriever,
        # delete_corrupted=args.delete_corrupted
    )
    
    # 打印总结
    print("\n" + "=" * 80)
    print("修复总结")
    print("=" * 80)
    print(f"成功修复: {results['success']} 个文件")
    print(f"修复失败: {results['failed']} 个文件")
    print(f"无需修复: {results['skipped']} 个文件")
    print("=" * 80)
    
    if results['failed'] > 0:
        print("\n[WARN] 部分文件修复失败，建议删除这些文件并重新生成")
        print("  可以运行以下命令删除损坏的文件：")
        print(f"  python fix_cache.py --cache-dir {args.cache_dir} --check-only")


if __name__ == "__main__":
    main()

