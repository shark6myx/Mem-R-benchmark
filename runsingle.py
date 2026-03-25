from memory_layer import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime

nltk.download('punkt_tab')
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None


class advancedMemAgent:
    # def __init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000):
    def __init__(self, model, backend, retrieve_k, temperature_c5):
        self.memory_system = AgenticMemorySystem(
            model_name='BAAI/bge-m3',
            llm_backend=backend,
            llm_model=model,

        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=None,

        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    # def retrieve_memory(self, content, k=10):
    #     return self.memory_system.find_related_memories_raw(content, k=k)
    def retrieve_memory(self, content, k=None, use_agentic: bool = False):
        if k is None:
            k = self.retrieve_k
            
        if use_agentic:
            # 使用新加入的 Agentic Decomposition 与 Reflection 分析闭环链路
            return self.memory_system.agentic_retrieve(content, k=k)
            
        # 默认：激活 GraphRAG 的社区感知检索！
        return self.memory_system.find_related_memories_with_community(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""

        # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,
                                                         response_format={"type": "json_schema", "json_schema": {
                                                             "name": "response",
                                                             "schema": {
                                                                 "type": "object",
                                                                 "properties": {
                                                                     "relevant_parts": {
                                                                         "type": "string",
                                                                     }
                                                                 },
                                                                 "required": ["relevant_parts"],
                                                                 "additionalProperties": False
                                                             },
                                                             "strict": True
                                                         }})
        # print("response:{}".format(response))
        return response

    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""

        # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,
                                                         response_format={"type": "json_schema", "json_schema": {
                                                             "name": "response",
                                                             "schema": {
                                                                 "type": "object",
                                                                 "properties": {
                                                                     "keywords": {
                                                                         "type": "string",
                                                                     }
                                                                 },
                                                                 "required": ["keywords"],
                                                                 "additionalProperties": False
                                                             },
                                                             "strict": True
                                                         }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def generate_chunk_context(self, session_full_text: str, turn_text: str) -> str:
        """
        Context Retrieval 核心方法：
        给定整个 Session 的上下文，提示 LLM 为单句对话生成其依赖的语境或解释。
        例如指代消解、意图说明等。
        """
        prompt = f"""You are a helpful assistant that provides context for isolated conversational turns.
Given the full conversation history (Session Full Text) and exactly ONE target isolated turn, write a short 1-2 sentence context explaining what the isolated turn refers to. For example, resolve any pronouns (he/she/it/this) to their specific entities, and clarify the topic being discussed.

Session Full Text:
{session_full_text}

Target Isolated Turn:
{turn_text}

Output ONLY your generated context as a JSON object:
{{
    "context": "Context explaining the turn..."
}}"""

        try:
            response = self.retriever_llm.llm.get_completion(
                prompt,
                response_format={
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "context_generation",
                        "schema": {
                            "type": "object",
                            "properties": {"context": {"type": "string"}},
                            "required": ["context"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                temperature=0.1
            )
            clean_resp = response.strip()
            if not clean_resp.startswith("{"):
                clean_resp = clean_resp[clean_resp.find("{"):]
            if not clean_resp.endswith("}"):
                clean_resp = clean_resp[:clean_resp.rfind("}")+1]
            return json.loads(clean_resp)["context"]
        except Exception as e:
            print(f"⚠ Failed to generate chunk context: {e}")
            return "General context"

    def answer_question(self, question: str, category: int, answer: str) -> str:
        """Generate answer for a question given the conversation context."""
        # 强制开启 Agentic Retrieval (包含逻辑拆解、反思令牌与逻辑闭环)
        # 传递完整的 question 而非 keywords，以便 AgenticDecomposer 获取完整上下文意图
        raw_context = self.retrieve_memory(question, k=self.retrieve_k, use_agentic=True)
        
        context = raw_context
        # print("context:", context)
        # context = self.retrieve_memory_llm(raw_context, question)
        # context = raw_context
        assert category in [1, 2, 3, 4, 5]
        user_prompt = f"""Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        temperature = 0.7
        if category == 5:  # adversial question, follow the initial paper.
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. {question} 

                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt, response_format={"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }}, temperature=temperature
        )
        # print(response)
        return response, user_prompt, raw_context


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None, ratio: float = 1.0,
                     backend: str = "sglang", temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000):
    """Evaluate the agent on the LoComo dataset.

    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
    """
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_ours_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = setup_logger(log_path)


    # 新增：创建结果保存目录（自动生成，不会重复）
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(os.path.dirname(__file__), "reranking_sample_results")  # 结果存在这个文件夹
    os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹，已有则不报错


    logger.info(f"Loading dataset from {dataset_path}")

    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")

    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio * 100:.1f}% of dataset)")

    # Store results
    results = []

    # 加速
    start_sample_idx = 9  # 👉 要跑第1个sample就写0，第2个写1，...，第10个写9
    samples = [samples[start_sample_idx]]  # 强制只保留1个sample
    logger.info(f"只运行 1 个 sample：原始索引 {start_sample_idx}（第 {start_sample_idx + 1} 个样本）")
    logger.info(
        f"Starting from sample {start_sample_idx + 1}/{len(samples) + start_sample_idx} (code index: {start_sample_idx})")

    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    # Evaluate each sample
    i = 0
    error_num = 0
    memories_dir = os.path.join(os.path.dirname(__file__), "cached_memories_advanced_{}_{}".format(backend, model))
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]
    for sample_idx, sample in enumerate(samples):
        # agent = advancedMemAgent(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)

        agent = advancedMemAgent(model, "openai", retrieve_k, temperature_c5)
        # Create memory cache filename based on sample and session indices
        # memory_cache_file = os.path.join(
        #     memories_dir,
        #     f"memory_cache_sample_{sample_idx}.pkl"
        # )
        # retriever_cache_file = os.path.join(
        #     memories_dir,
        #     f"retriever_cache_sample_{sample_idx}.pkl"
        # )
        # retriever_cache_embeddings_file = os.path.join(
        #     memories_dir,
        #     f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        # )

        # 关键：计算原样本索引（start_sample_idx=8 + 新索引0/1 → 8/9，对应原样本9/10）
        original_sample_idx = start_sample_idx + sample_idx
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_sample_{original_sample_idx}.pkl"  # 原样本9→8.pkl，原样本10→9.pkl
        )
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{original_sample_idx}.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_embeddings_sample_{original_sample_idx}.npy"
        )
        # ── 持久化记忆加载 / 重建逻辑 ─────────────────────────────────────
        # 策略：检测缓存是否为含新字段（community_id）的新格式
        #       新格式 → 直接加载并恢复完整 GraphRAG 状态
        #       旧格式或无缓存 → 重新构建并保存
        # ─────────────────────────────────────────────────────────────────
        def _is_new_format(mem_dict: dict) -> bool:
            """检查 pickle 里的 MemoryNote 是否含新字段"""
            if not mem_dict:
                return False
            sample = next(iter(mem_dict.values()))
            return hasattr(sample, 'community_id') and hasattr(sample, 'id_based_links')

        cache_loaded = False
        if os.path.exists(memory_cache_file):
            try:
                with open(memory_cache_file, 'rb') as f:
                    cached_memories = pickle.load(f)

                if _is_new_format(cached_memories):
                    # ── 新格式：完整恢复 ──────────────────────────────
                    logger.info(f"✅ 检测到新格式缓存，加载 {len(cached_memories)} 条记忆...")
                    agent.memory_system.memories = cached_memories
                    agent.memory_system.note_total_count = len(cached_memories)

                    # 恢复检索器（优先用完整缓存，其次只用 embeddings）
                    if os.path.exists(retriever_cache_file):
                        agent.memory_system.retriever = agent.memory_system.retriever.load(
                            retriever_cache_file, retriever_cache_embeddings_file
                        )
                    elif os.path.exists(retriever_cache_embeddings_file):
                        agent.memory_system.retriever = \
                            agent.memory_system.retriever.load_from_local_memory(
                                cached_memories, 'BAAI/bge-m3',
                                agent.memory_system.retriever_alpha,
                                embeddings_cache_file=retriever_cache_embeddings_file
                            )
                    else:
                        agent.memory_system.retriever = \
                            agent.memory_system.retriever.load_from_local_memory(
                                cached_memories, 'BAAI/bge-m3',
                                agent.memory_system.retriever_alpha
                            )

                    # 恢复 GraphRAG 图边 + 社区状态
                    graph_cache_dir = os.path.join(memories_dir, f"graph_state_sample_{original_sample_idx}")
                    if os.path.exists(graph_cache_dir):
                        agent.memory_system.load_graph(graph_cache_dir)
                        logger.info(f"✅ GraphRAG 图状态已恢复（"
                                    f"{len(agent.memory_system.graph_edges)} 条边，"
                                    f"{len(agent.memory_system.communities)} 个社区）")
                    else:
                        # 没有图状态缓存，基于已有记忆重建社区
                        logger.info("⚠ 无图状态缓存，基于现有记忆重建社区...")
                        agent.memory_system.rebuild_communities()

                    cache_loaded = True
                else:
                    logger.info("⚠ 检测到旧格式缓存（缺少新字段），忽略，将重新构建记忆")
            except Exception as e:
                logger.info(f"⚠ 加载缓存失败：{e}，将重新构建记忆")

        if not cache_loaded:
            # ── 重新构建：逐条 add_note，触发实时演化 + 社区聚类 ─────
            logger.info(f"🔄 重新构建记忆（sample {original_sample_idx}）...")
            for _, turns in sample.conversation.sessions.items():
                
                # Context Retrieval 步骤 1：构建全局文档视角 (Global Session View)
                session_full_text = "\n".join([f"Speaker {t.speaker} says : {t.text}" for t in turns.turns])
                
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + " says : " + turn.text
                    
                    # Context Retrieval 步骤 2：局部块上下文生成 (Chunk Context Generation)
                    # 提示 LLM 根据全局上下文为单个孤立句子生成背景解释
                    chunk_context = agent.generate_chunk_context(session_full_text, conversation_tmp)
                    
                    # Context Retrieval 步骤 3：内容融合与向量化
                    enriched_content = f"Context: {chunk_context}\nContent: {conversation_tmp}"
                    
                    agent.add_memory(enriched_content, time=turn_datatime)

            # 同步计数（add_note 已自增，这里做最终校正）
            agent.memory_system.note_total_count = len(agent.memory_system.memories)

            # 最后触发一次完整社区重建（确保所有记忆都已纳入）
            if len(agent.memory_system.memories) >= 2:
                agent.memory_system.rebuild_communities()

            # ── 保存全量状态（memories + retriever + graph）────────────
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            graph_cache_dir = os.path.join(memories_dir, f"graph_state_sample_{original_sample_idx}")
            agent.memory_system.save_graph(graph_cache_dir)
            logger.info(f"✅ 已缓存 {len(memories_to_cache)} 条记忆 + GraphRAG 图状态 → {memories_dir}")
        # ─────────────────────────────────────────────────────────────────


        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(samples)}")

        # 新增：初始化列表，存储当前Sample的所有QA结果
        sample_results = []

        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                # Generate prediction
                prediction, user_prompt, raw_context = agent.answer_question(qa.question, qa.category, qa.final_answer)
                try:
                    prediction = json.loads(prediction)["answer"]
                except:
                    # 增强的JSON解析和兜底逻辑
                    import re
                    # 1. 尝试清洗 Markdown 标记
                    clean_prediction = prediction.strip()
                    if clean_prediction.startswith("```json"):
                        clean_prediction = clean_prediction[7:]
                    elif clean_prediction.startswith("```"):
                        clean_prediction = clean_prediction[3:]
                    if clean_prediction.endswith("```"):
                        clean_prediction = clean_prediction[:-3]
                    clean_prediction = clean_prediction.strip()

                    try:
                        # 2. 尝试解析清洗后的 JSON
                        prediction = json.loads(clean_prediction)["answer"]
                    except:
                        # 3. 尝试正则提取
                        match = re.search(r'"answer":\s*"(.*?)"', clean_prediction)
                        if match:
                            prediction = match.group(1)
                        else:
                            # 4. 彻底失败，使用清洗后的文本作为答案
                            prediction = clean_prediction
                            logger.info(f"Failed to parse prediction as JSON: {prediction}")
                            error_num += 1
                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"User Prompt: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Raw Context: {raw_context}")

                # Calculate metrics
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                # Store individual result
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics
                }
                results.append(result)
                # 新增：将当前QA结果加入当前Sample的列表
                sample_results.append(result)
                # Log progress
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")

        # 新增：保存当前Sample的独立结果文件
        original_sample_idx = start_sample_idx + sample_idx  # 用原始索引命名（避免分多次跑时文件名重复）
        sample_output_file = os.path.join(output_dir, f"reranking_result_sample_{original_sample_idx}.json")
        with open(sample_output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "sample_idx": original_sample_idx,  # 样本原始索引（0-9）
                "total_questions": len(sample_results),  # 该样本的问题数
                "individual_results": sample_results  # 该样本的所有QA详细结果
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 已保存Sample {original_sample_idx} 结果到：{sample_output_file}")


    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    # Prepare final results
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        # "individual_results": results
    }
    logger.info(f"Error number: {error_num}")

    # 自动生成：base_sample 文件夹下的对应文件名
    if not output_path:  # 如果没手动指定 --output，自动生成路径

        original_sample_idx = start_sample_idx  # 当前 sample 的原始索引（0-9）

        # 文件名格式：result_sample_XXX_final.json（XXX是原始索引）
        output_filename = f"reranking_result_sample_{original_sample_idx}_final.json"
        # 路径：base_sample 文件夹 + 自动生成的文件名
        output_path = os.path.join(output_dir, output_filename)
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} questions ({count / total_questions * 100:.1f}%)")

    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="sglang",
                        help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=40,
                        help="Retrieve k")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None

    evaluate_dataset(dataset_path, args.model, output_path, args.ratio, args.backend, args.temperature_c5,
                     args.retrieve_k, args.sglang_host, args.sglang_port)


if __name__ == "__main__":
    main()