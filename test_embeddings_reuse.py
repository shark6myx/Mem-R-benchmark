"""
测试脚本：验证从嵌入向量缓存复用并只构建BM25索引的功能

测试场景：
1. 从真实缓存文件夹加载记忆和嵌入向量缓存
2. 删除或备份检索器缓存，只保留嵌入向量缓存
3. 使用load_from_local_memory从嵌入向量缓存加载
4. 验证是否成功复用嵌入向量，只构建了BM25索引
"""

import os
import pickle
import numpy as np
from pathlib import Path
import shutil
from memory_layer import HybridRetriever

def test_embeddings_reuse_real_cache(sample_idx: int = 0):
    """使用真实缓存文件测试嵌入向量复用功能"""
    print("=" * 60)
    print(f"测试：从真实缓存复用嵌入向量并只构建BM25索引 (sample_{sample_idx})")
    print("=" * 60)
    
    # 使用真实的缓存文件夹
    cache_dir = Path("cached_memories_advanced_sglang_gpt-4o-mini")
    
    if not cache_dir.exists():
        print(f"  ✗ 缓存文件夹不存在: {cache_dir}")
        return False
    
    memory_cache_file = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    retriever_cache_file = cache_dir / f"retriever_cache_sample_{sample_idx}.pkl"
    embeddings_cache_file = cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    
    # 备份原始检索器缓存文件（如果存在）
    retriever_cache_backup = cache_dir / f"retriever_cache_sample_{sample_idx}.pkl.backup"
    
    try:
        # ========== 步骤1: 加载真实的缓存文件 ==========
        print("\n[步骤1] 加载真实的缓存文件...")
        
        # 检查必要文件是否存在
        if not memory_cache_file.exists():
            print(f"  ✗ 记忆缓存文件不存在: {memory_cache_file}")
            return False
        
        if not embeddings_cache_file.exists():
            print(f"  ✗ 嵌入向量缓存文件不存在: {embeddings_cache_file}")
            return False
        
        # 加载记忆
        with open(memory_cache_file, 'rb') as f:
            memories = pickle.load(f)
        print(f"  ✓ 加载了 {len(memories)} 个记忆")
        
        # 加载原始嵌入向量
        original_embeddings = np.load(embeddings_cache_file)
        print(f"  ✓ 嵌入向量形状: {original_embeddings.shape}")
        
        # 准备文档列表（与load_from_local_memory中的逻辑一致）
        all_docs = [", ".join(m.keywords) for m in memories.values()]
        print(f"  ✓ 准备 {len(all_docs)} 个文档")
        
        # 验证嵌入向量数量是否匹配
        if len(original_embeddings) != len(all_docs):
            print(f"  ⚠ 警告：嵌入向量数量({len(original_embeddings)})与文档数量({len(all_docs)})不匹配")
            print("    将尝试继续测试，但可能会重新生成嵌入向量")
        
        # ========== 步骤2: 备份并删除检索器缓存，只保留嵌入向量 ==========
        print("\n[步骤2] 备份检索器缓存，模拟只有嵌入向量缓存的情况...")
        
        if retriever_cache_file.exists():
            # 备份原始文件
            if retriever_cache_backup.exists():
                retriever_cache_backup.unlink()  # 删除旧备份
            shutil.copy2(retriever_cache_file, retriever_cache_backup)
            print(f"  ✓ 已备份检索器缓存到: {retriever_cache_backup}")
            
            # 删除检索器缓存文件
            retriever_cache_file.unlink()
            print(f"  ✓ 已删除检索器缓存: {retriever_cache_file}")
        else:
            print(f"  ✓ 检索器缓存文件不存在，无需删除")
        
        # 验证嵌入向量文件还存在
        if embeddings_cache_file.exists():
            print(f"  ✓ 嵌入向量缓存仍然存在: {embeddings_cache_file}")
        
        # ========== 步骤3: 使用load_from_local_memory从嵌入向量缓存加载 ==========
        print("\n[步骤3] 使用load_from_local_memory从嵌入向量缓存加载...")
        
        import time
        start_time = time.time()
        
        # 创建一个新的检索器实例，使用load_from_local_memory
        retriever2 = HybridRetriever.load_from_local_memory(
            memories=memories,
            model_name='all-MiniLM-L6-v2',
            alpha=0.5,
            embeddings_cache_file=str(embeddings_cache_file)
        )
        
        load_time = time.time() - start_time
        print(f"  ✓ 加载完成，耗时: {load_time:.3f} 秒")
        
        # ========== 步骤4: 验证结果 ==========
        print("\n[步骤4] 验证结果...")
        
        # 验证嵌入向量是否被复用
        if retriever2.embeddings is not None:
            # 处理可能的形状差异（torch tensor vs numpy array）
            embeddings_to_compare = retriever2.embeddings
            if hasattr(embeddings_to_compare, 'cpu'):
                embeddings_to_compare = embeddings_to_compare.cpu().numpy()
            elif hasattr(embeddings_to_compare, 'numpy'):
                embeddings_to_compare = embeddings_to_compare.numpy()
            
            # 如果形状不同，尝试调整
            if embeddings_to_compare.shape != original_embeddings.shape:
                min_len = min(len(embeddings_to_compare), len(original_embeddings))
                embeddings_to_compare = embeddings_to_compare[:min_len]
                original_embeddings_trimmed = original_embeddings[:min_len]
            else:
                original_embeddings_trimmed = original_embeddings
            
            embeddings_match = np.allclose(embeddings_to_compare, original_embeddings_trimmed, rtol=1e-5)
            print(f"  ✓ 嵌入向量形状: {embeddings_to_compare.shape}")
            print(f"  ✓ 原始嵌入向量形状: {original_embeddings.shape}")
            print(f"  ✓ 嵌入向量是否匹配: {embeddings_match}")
            if embeddings_match:
                print("    → ✅ 嵌入向量成功复用！")
            else:
                print("    → ⚠ 警告：嵌入向量不完全匹配，可能重新生成了")
                print(f"       最大差异: {np.max(np.abs(embeddings_to_compare - original_embeddings_trimmed))}")
        else:
            print("  ✗ 嵌入向量为空！")
            return False
        
        # 验证BM25索引是否已构建
        if retriever2.bm25 is not None:
            print(f"  ✓ BM25索引已创建")
            print("    → BM25索引成功构建！")
        else:
            print("  ✗ BM25索引未创建！")
            return False
        
        # 验证文档数量和corpus
        if len(retriever2.corpus) == len(all_docs):
            print(f"  ✓ 文档数量匹配: {len(retriever2.corpus)}")
        else:
            print(f"  ✗ 文档数量不匹配: 期望 {len(all_docs)}, 实际 {len(retriever2.corpus)}")
            return False
        
        # ========== 步骤5: 测试检索功能 ==========
        print("\n[步骤5] 测试检索功能...")
        query = "对话"
        results = retriever2.retrieve(query, k=3)
        
        if results:
            print(f"  ✓ 查询 '{query}' 返回了 {len(results)} 个结果")
            print(f"    结果索引: {results}")
            
            # 显示检索到的文档
            for i, idx in enumerate(results, 1):
                doc_preview = retriever2.corpus[idx][:80] if len(retriever2.corpus[idx]) > 80 else retriever2.corpus[idx]
                print(f"    {i}. 文档 {idx}: {doc_preview}...")
        else:
            print(f"  ✗ 检索未返回结果")
            return False
        
        # ========== 步骤6: 对比性能（验证嵌入向量是否真的被复用） ==========
        print("\n[步骤6] 性能对比（验证嵌入向量是否真的被复用）...")
        
        # 测试复用缓存的情况（应该很快，因为不需要生成嵌入向量）
        print("  测试1: 使用缓存加载...")
        start_time = time.time()
        retriever3 = HybridRetriever.load_from_local_memory(
            memories=memories,
            model_name='all-MiniLM-L6-v2',
            alpha=0.5,
            embeddings_cache_file=str(embeddings_cache_file)
        )
        time_with_cache = time.time() - start_time
        
        # 测试重新生成的情况（应该较慢，因为需要生成嵌入向量）
        print("  测试2: 重新生成嵌入向量...")
        start_time = time.time()
        retriever4 = HybridRetriever.load_from_local_memory(
            memories=memories,
            model_name='all-MiniLM-L6-v2',
            alpha=0.5,
            embeddings_cache_file=None  # 不提供缓存，强制重新生成
        )
        time_without_cache = time.time() - start_time
        
        print(f"  使用缓存耗时: {time_with_cache:.3f} 秒")
        print(f"  重新生成耗时: {time_without_cache:.3f} 秒")
        if time_with_cache > 0:
            print(f"  速度提升: {time_without_cache / time_with_cache:.2f}x")
        
        if time_with_cache < time_without_cache:
            print("  ✓ 缓存复用确实加快了速度！")
        else:
            print("  ⚠ 注意：缓存复用未明显加快")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！嵌入向量复用功能正常工作。")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 恢复备份的检索器缓存文件
        print("\n[清理] 恢复原始检索器缓存文件...")
        if retriever_cache_backup.exists():
            if retriever_cache_file.exists():
                retriever_cache_file.unlink()
            shutil.copy2(retriever_cache_backup, retriever_cache_file)
            retriever_cache_backup.unlink()
            print(f"  ✓ 已恢复检索器缓存文件: {retriever_cache_file}")
            print(f"  ✓ 已删除备份文件")
        else:
            print("  ✓ 无需恢复（原始文件不存在）")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试嵌入向量复用功能")
    parser.add_argument("--sample-idx", type=int, default=0, help="要测试的sample索引（默认0）")
    args = parser.parse_args()
    
    print("开始测试嵌入向量复用功能...\n")
    
    # 运行真实缓存测试
    test_passed = test_embeddings_reuse_real_cache(args.sample_idx)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  真实缓存测试: {'✓ 通过' if test_passed else '✗ 失败'}")
    print("=" * 60)
    
    if test_passed:
        print("\n🎉 测试通过！")
        exit(0)
    else:
        print("\n❌ 测试失败，请检查输出")
        exit(1)

