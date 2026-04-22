from typing import Optional, Dict, List, Literal, Any, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, retry_if_exception_type
import openai
import json
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
from litellm import completion
import requests
import json as json_lib
import time
import torch
import sys
import re

from evidence_program import AnswerStyle, EvidenceProgram, ExecutionTrace, ProgramType
from model_cache_utils import load_sentence_transformer
from reranker_utils import load_reranker
from retrieval_types import EvidenceSubgraph, RetrievalEngine, RetrievalResult

MEMORY_TEXT_SCHEMA_VERSION = 2
RETRIEVER_CACHE_VERSION = 2
GRAPH_CACHE_VERSION = 2


def _should_retry_openai_exception(exc: BaseException) -> bool:
    if isinstance(exc, openai.AuthenticationError):
        return False
    if isinstance(exc, openai.NotFoundError):
        return False
    if isinstance(exc, openai.BadRequestError):
        return False
    return isinstance(exc, (openai.APIConnectionError, openai.APIError))

class BaseLLMController(ABC):
    """
    LLM控制器的抽象基类
    
    定义了所有LLM控制器必须实现的接口方法
    """
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """
        从LLM获取完成响应（抽象方法）
        
        参数:
            prompt: 输入的提示文本
            
        返回:
            LLM生成的响应文本
        """
        pass

class OpenAIController(BaseLLMController):
    """
    OpenAI API控制器
    
    用于调用OpenAI接口进行文本生成
    """
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化OpenAI控制器
        
        参数:
            model: 使用的模型名称，默认为"gpt-4"
            api_key: OpenAI API密钥，如果为None则使用默认密钥
        """
        try:
            from openai import OpenAI
            self.model = model
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
            resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
            if resolved_base_url:
                self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
            else:
                self.client = OpenAI(api_key=resolved_api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    @retry(
        stop=stop_after_attempt(5),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=2, max=5),  # 重试间隔：指数退避，最小2秒，最大5秒
        retry=retry_if_exception(_should_retry_openai_exception),
    )
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        """
        从OpenAI获取文本完成响应
        
        参数:
            prompt: 输入的提示文本
            response_format: 响应格式配置字典
            temperature: 生成温度，控制随机性，默认为0.7
            
        返回:
            LLM生成的响应内容字符串
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    """
    Ollama本地LLM控制器
    
    用于调用本地部署的Ollama模型
    """
    def __init__(self, model: str = "llama2"):
        """
        初始化Ollama控制器
        
        参数:
            model: 使用的模型名称，默认为"llama2"
        """
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """
        根据JSON Schema类型生成空值
        
        参数:
            schema_type: JSON Schema类型（"array", "string", "object", "number", "boolean"）
            schema_items: 数组项的schema定义（可选）
            
        返回:
            对应类型的空值
        """
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class SGLangController(BaseLLMController):
    """
    SGLang服务器控制器
    
    用于调用SGLang服务器进行文本生成
    """
    def __init__(self, model: str = "llama2", sglang_host: str = "http://localhost", sglang_port: int = 30000):
        """
        初始化SGLang控制器
        
        参数:
            model: 使用的模型名称，默认为"llama2"
            sglang_host: SGLang服务器主机地址，默认为"http://localhost"
            sglang_port: SGLang服务器端口，默认为30000
        """
        self.model = model
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.base_url = f"{sglang_host}:{sglang_port}"
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """
        根据JSON Schema类型生成空值
        
        参数:
            schema_type: JSON Schema类型（"array", "string", "object", "number", "integer", "boolean"）
            schema_items: 数组项的schema定义（可选）
            
        返回:
            对应类型的空值
        """
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number" or schema_type == "integer":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """
        根据响应格式生成空的响应结构
        
        参数:
            response_format: 响应格式配置字典
            
        返回:
            符合schema的空响应字典
        """
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        """
        从SGLang服务器获取文本完成响应
        
        参数:
            prompt: 输入的提示文本
            response_format: 响应格式配置字典
            temperature: 生成温度，控制随机性，默认为0.7
            
        返回:
            LLM生成的响应内容字符串，如果出错则返回空响应JSON
        """
        try:
            # 从response_format中提取JSON schema并转换为字符串格式
            json_schema = response_format.get("json_schema", {}).get("schema", {})
            json_schema_str = json.dumps(json_schema)
            
            # 准备SGLang请求的正确格式
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "json_schema": json_schema_str  # SGLang期望JSON schema为字符串格式
                }
            }
            
            # 向SGLang服务器发送请求
            response = requests.post(
                f"{self.base_url}/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # SGLang返回的生成文本在'text'字段中
                generated_text = result.get("text", "")
                return generated_text
            else:
                print(f"SGLang server returned status {response.status_code}: {response.text}")
                raise Exception(f"SGLang server error: {response.status_code}")
                
        except Exception as e:
            print(f"SGLang completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LiteLLMController(BaseLLMController):
    """
    LiteLLM通用控制器
    
    用于统一访问包括Ollama和SGLang在内的各种LLM
    """
    def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        """
        初始化LiteLLM控制器
        
        参数:
            model: 使用的模型名称
            api_base: API基础URL（可选）
            api_key: API密钥（可选），默认值为"EMPTY"
        """
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        """
        根据JSON Schema类型生成空值
        
        参数:
            schema_type: JSON Schema类型（"array", "string", "object", "number", "boolean"）
            schema_items: 数组项的schema定义（可选）
            
        返回:
            对应类型的空值
        """
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        """
        根据响应格式生成空的响应结构
        
        参数:
            response_format: 响应格式配置字典
            
        返回:
            符合schema的空响应字典
        """
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        """
        从LiteLLM获取文本完成响应
        
        参数:
            prompt: 输入的提示文本
            response_format: 响应格式配置字典
            temperature: 生成温度，控制随机性，默认为0.7
            
        返回:
            LLM生成的响应内容字符串，如果出错则返回空响应JSON
        """
        try:
            # 准备完成请求的参数
            completion_args = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": response_format,
                "temperature": temperature
            }
            
            # 如果提供了API基础URL和密钥，则添加到参数中
            if self.api_base:
                completion_args["api_base"] = self.api_base
            if self.api_key:
                completion_args["api_key"] = self.api_key
                
            response = completion(**completion_args)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LiteLLM completion error: {e}")
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """
    LLM控制器包装类
    
    用于生成记忆元数据的LLM控制器，支持多种后端
    """
    def __init__(self, 
                 backend: Literal["openai", "ollama", "sglang", "vllm"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 ):
        """
        初始化LLM控制器
        
        参数:
            backend: 后端类型，可选"openai", "ollama", "sglang"，默认为"openai"
            model: 使用的模型名称，默认为"gpt-4"
            api_key: API密钥（可选）
            api_base: API基础URL（可选）
        """
        if backend == "openai":
            self.llm = OpenAIController(model=model, api_key=api_key, base_url=api_base)
        elif backend == "vllm":
            if not api_base:
                raise ValueError("vllm backend requires api_base like http://127.0.0.1:8000/v1")
            self.llm = OpenAIController(model=model, api_key=api_key or "EMPTY", base_url=api_base)
        elif backend == "ollama":
            # 使用LiteLLM控制Ollama并支持JSON输出
            ollama_model = f"ollama/{model}" if not model.startswith("ollama/") else model
            self.llm = LiteLLMController(
                model=ollama_model, 
                api_base="http://localhost:11434", 
                api_key="EMPTY"
            )
        elif backend == "sglang":
            self.llm = SGLangController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'vllm', 'ollama', or 'sglang'")

class MemoryNote:
    """
    基本记忆单元类
    
    包含内容及其元数据（关键词、链接、重要性评分等）
    """
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None, 
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):
        """
        初始化记忆笔记
        
        参数:
            content: 记忆内容文本
            id: 记忆唯一标识符，如果为None则自动生成UUID
            keywords: 关键词列表
            links: 关联的其他记忆链接
            importance_score: 重要性评分，默认为1.0
            retrieval_count: 检索次数计数
            timestamp: 创建时间戳
            last_accessed: 最后访问时间
            context: 上下文信息
            evolution_history: 演化历史记录
            category: 分类类别
            tags: 标签列表
            llm_controller: LLM控制器，用于自动生成元数据
        """
        self.content = content
        
        # 如果未提供元数据且LLM控制器可用，则使用LLM生成元数据
        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            print("analysis", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]
        
        # 为可选参数设置默认值
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # 处理上下文，可以是字符串或列表
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)  # 将列表转换为字符串

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []
        self.text_schema_version = MEMORY_TEXT_SCHEMA_VERSION
        # GraphRAG: community membership and stable id-based links
        self.community_id: Optional[str] = None
        self.id_based_links: List[str] = []  # stores note.id strings (not position indices)
        self.normalize_metadata()

    @staticmethod
    def _extract_speaker_terms(content: str) -> set:
        names = set()
        for match in re.finditer(r"Speaker\s+([A-Za-z][A-Za-z0-9_-]*)\s+says", content or "", flags=re.IGNORECASE):
            names.add(match.group(1).lower())
        return names

    @staticmethod
    def _tokenize_terms(text: str) -> List[str]:
        return re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", (text or "").lower())

    @classmethod
    def _trim_text_to_budget(cls, text: str, max_chars: int) -> str:
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        clipped = text[:max_chars].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0].rstrip()
        return clipped.rstrip(" ,;:-")

    @classmethod
    def _compress_context_text(
        cls,
        context: Union[str, List[str], None],
        content: str = "",
        max_sentences: int = 2,
        max_chars: int = 240,
    ) -> str:
        if isinstance(context, list):
            raw = " ".join(str(item) for item in context if item)
        else:
            raw = str(context or "")
        raw = raw.strip()
        if not raw:
            return "General"

        raw = re.sub(r"\s*\|\s*Context:\s*", ". ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\bContext:\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s+", " ", raw).strip(" |")
        raw = re.sub(r"([.!?])\s*\.", r"\1", raw)
        if not raw:
            return "General"

        fragments = [frag.strip(" |") for frag in re.split(r"(?<=[.!?])\s+|\s+\|\s+", raw) if frag.strip(" |")]
        if not fragments:
            fragments = [raw]

        sentences: List[str] = []
        seen = set()
        for fragment in fragments:
            cleaned = re.sub(r"\s+", " ", fragment).strip(" ,;:-")
            norm = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
            if not cleaned or not norm or norm in seen:
                continue
            seen.add(norm)
            sentences.append(cleaned)

        if not sentences:
            return "General"

        stopwords = {
            "speaker", "says", "context", "content", "general", "about", "into", "from", "with", "that",
            "this", "they", "them", "their", "there", "where", "when", "which", "while", "after", "before",
            "because", "would", "could", "should", "have", "been", "were", "being", "also", "just", "like",
        }
        speaker_terms = cls._extract_speaker_terms(content)
        content_terms = {
            term for term in cls._tokenize_terms(content)
            if len(term) >= 3 and term not in stopwords and term not in speaker_terms
        }
        sentence_terms = []
        for sentence in sentences:
            terms = {
                term for term in cls._tokenize_terms(sentence)
                if len(term) >= 3 and term not in stopwords
            }
            sentence_terms.append(terms)

        selected_indices = [0]
        used_terms = set(sentence_terms[0])
        while len(selected_indices) < min(max_sentences, len(sentences)):
            best_idx = None
            best_score = -1.0
            for idx in range(1, len(sentences)):
                if idx in selected_indices:
                    continue
                overlap = len(sentence_terms[idx] & content_terms)
                novelty = len(sentence_terms[idx] - used_terms)
                score = overlap * 3.0 + novelty + max(0.0, 0.25 - (idx * 0.03))
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                break
            candidate_text = " ".join(sentences[i] for i in selected_indices + [best_idx]).strip()
            if len(candidate_text) > max_chars:
                break
            selected_indices.append(best_idx)
            used_terms |= sentence_terms[best_idx]

        selected_text = " ".join(sentences[i] for i in selected_indices).strip()
        selected_text = cls._trim_text_to_budget(selected_text, max_chars)
        selected_text = re.sub(r"([.!?])\1+", r"\1", selected_text)
        return selected_text or "General"

    @classmethod
    def _normalize_keywords(
        cls,
        keywords: Optional[List[str]],
        content: str = "",
        max_items: int = 6,
    ) -> List[str]:
        stopwords = {
            "speaker", "says", "context", "content", "general", "today", "tomorrow", "yesterday",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december",
        }
        speaker_terms = cls._extract_speaker_terms(content)
        normalized: List[str] = []
        seen = set()
        for keyword in keywords or []:
            cleaned = re.sub(r"\s+", " ", str(keyword or "")).strip(" ,;:-")
            if not cleaned:
                continue
            norm = cleaned.lower()
            if norm in seen or norm in stopwords or norm in speaker_terms:
                continue
            if not re.search(r"[A-Za-z]", cleaned):
                continue
            seen.add(norm)
            normalized.append(cleaned)
            if len(normalized) >= max_items:
                break
        return normalized

    @classmethod
    def _normalize_tags(
        cls,
        tags: Optional[List[str]],
        keywords: Optional[List[str]] = None,
        max_items: int = 8,
    ) -> List[str]:
        generic_tags = {
            "conversation", "personal_interests", "personal interests", "social interaction",
            "shared appreciation", "general", "context", "content", "discussion", "communication",
        }
        keyword_norms = {str(item).strip().lower() for item in keywords or [] if str(item).strip()}
        normalized: List[str] = []
        seen = set()
        for tag in tags or []:
            cleaned = re.sub(r"\s+", " ", str(tag or "")).strip(" ,;:-")
            if not cleaned:
                continue
            norm = cleaned.lower()
            if norm in seen or norm in generic_tags or norm in keyword_norms or len(norm) <= 2:
                continue
            seen.add(norm)
            normalized.append(cleaned)
            if len(normalized) >= max_items:
                break
        return normalized

    def normalize_metadata(self) -> None:
        self.context = self._compress_context_text(self.context, self.content)
        self.keywords = self._normalize_keywords(self.keywords, self.content)
        self.tags = self._normalize_tags(self.tags, self.keywords)

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:
        """
        分析内容以提取关键词、上下文和其他元数据
        
        参数:
            content: 待分析的内容文本
            llm_controller: LLM控制器实例
            
        返回:
            包含keywords、context、tags的字典
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        response: Optional[str] = None
        try:
            response = llm_controller.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                }
            })

            try:
                # 清理响应，以防有多余的文本
                response_cleaned = response.strip()
                # 尝试从其他文本中提取JSON内容
                if not response_cleaned.startswith('{'):
                    start_idx = response_cleaned.find('{')
                    if start_idx != -1:
                        response_cleaned = response_cleaned[start_idx:]
                if not response_cleaned.endswith('}'):
                    end_idx = response_cleaned.rfind('}')
                    if end_idx != -1:
                        response_cleaned = response_cleaned[:end_idx+1]

                analysis = json.loads(response_cleaned)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in analyze_content: {e}")
                print(f"Raw response: {response}")
                analysis = {
                    "keywords": [],
                    "context": "General",
                    "tags": []
                }

            return analysis

        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
            if response is not None:
                print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }

@dataclass
class CommunitySummary:
    """
    GraphRAG 社区摘要单元
    
    代表一组语义相关的 MemoryNote 聚类，由 LLM 生成高层摘要，
    供 Global Search 使用。
    """
    community_id: str
    level: int                              # 0=细粒度层级
    member_note_ids: List[str]             # 隶属 note 的 id 列表
    title: str = ""                         # LLM 生成的主题标题（≤20字）
    summary: str = ""                       # LLM 生成的综合摘要（≤200字）
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None         # summary 的向量（np.ndarray）
    updated_at: str = ""
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化（不含 embedding，embedding 单独存）"""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "member_note_ids": self.member_note_ids,
            "title": self.title,
            "summary": self.summary,
            "keywords": self.keywords,
            "updated_at": self.updated_at,
            "evolution_history": self.evolution_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CommunitySummary":
        return cls(
            community_id=d["community_id"],
            level=d.get("level", 0),
            member_note_ids=d.get("member_note_ids", []),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            keywords=d.get("keywords", []),
            updated_at=d.get("updated_at", ""),
            evolution_history=d.get("evolution_history", []),
        )


class HybridRetriever:
    """
    语义检索系统
    
    使用 Dense 向量的语义检索器
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', dense_weight: float = 0.75):
        """
        初始化语义检索器
        
        参数:
            model_name: 使用的SentenceTransformer模型名称
        """
        self.model_name = model_name
        self.model = load_sentence_transformer(model_name)
        self.corpus = []
        self.dense_corpus = []
        self.embeddings = None
        self.document_ids = {}  # 文档内容到索引的映射
        self.dense_weight = float(dense_weight)
        
        # ── 稀疏检索 (BM25) ──────────────────────────
        self.bm25 = None
        self.tokenized_corpus = []

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """尽量稳定地进行 BM25 分词，缺失 jieba 时回退到空格切词。"""
        try:
            import jieba
            return list(jieba.cut(text))
        except Exception:
            return (text or "").lower().split()

    def _min_max_norm_array(self, scores: np.ndarray) -> np.ndarray:
        if scores is None or len(scores) == 0:
            return np.array([])
        scores = np.asarray(scores, dtype=float)
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        if max_v - min_v < 1e-8:
            return np.ones_like(scores, dtype=float)
        return (scores - min_v) / (max_v - min_v)

    def get_hybrid_scores(self, query: str, query_embedding: Optional[np.ndarray] = None, dense_weight: Optional[float] = None) -> np.ndarray:
        """
        混合 Dense + BM25 分数。
        如果 BM25 不可用，则自动退化为 Dense。
        """
        if self.embeddings is None or len(self.corpus) == 0:
            return np.array([])

        if query_embedding is None:
            query_embedding = self.model.encode([query])
            if hasattr(query_embedding, 'numpy'):
                query_embedding = query_embedding.numpy()
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding[0]

        dense_scores = self.get_scores_by_emb(query_embedding)
        if len(dense_scores) == 0:
            return np.array([])

        dense_norm = self._min_max_norm_array(dense_scores)

        if self.bm25 is None or not self.tokenized_corpus:
            return dense_norm

        query_tokens = self._tokenize_for_bm25(query)
        bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        bm25_norm = self._min_max_norm_array(bm25_scores)

        w_dense = self.dense_weight if dense_weight is None else float(dense_weight)
        w_dense = max(0.0, min(1.0, w_dense))
        w_bm25 = 1.0 - w_dense
        return w_dense * dense_norm + w_bm25 * bm25_norm
        
    
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """
        将检索器状态保存到磁盘
        
        参数:
            retriever_cache_file: 检索器缓存文件路径
            retriever_cache_embeddings_file: 嵌入向量缓存文件路径
        """
        
        # 使用numpy保存嵌入向量
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
            
        # 使用pickle保存其他所有内容
        state = {
            'cache_version': RETRIEVER_CACHE_VERSION,
            'corpus': self.corpus,
            'dense_corpus': self.dense_corpus,
            'document_ids': self.document_ids,
            'model_name': 'BAAI/bge-m3',  # 模型名称的默认值
            'dense_weight': self.dense_weight,
        }
        
        # 尝试获取实际的模型名称（如果可能）
        try:
            state['model_name'] = self.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            pass
        state['model_name'] = self.model_name
            
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """
        从磁盘加载检索器状态
        
        参数:
            retriever_cache_file: 检索器缓存文件路径
            retriever_cache_embeddings_file: 嵌入向量缓存文件路径
            
        返回:
            加载后的检索器实例
        """
        # 加载pickle序列化的状态
        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)
        cache_version = state.get('cache_version', 0)
        if cache_version != RETRIEVER_CACHE_VERSION:
            raise ValueError(
                f"Retriever cache version mismatch: expected {RETRIEVER_CACHE_VERSION}, got {cache_version}"
            )
        
        # 使用默认值，防止缺少键的情况
        model_name = state.get('model_name', 'BAAI/bge-m3')
            
        # 创建新实例
        dense_weight = state.get('dense_weight', 0.70)
        retriever = cls(model_name=model_name, dense_weight=dense_weight)
        retriever.corpus = state.get('corpus', [])
        retriever.dense_corpus = state.get('dense_corpus', list(retriever.corpus))
        retriever.document_ids = state.get('document_ids', {})
        if len(retriever.dense_corpus) != len(retriever.corpus):
            raise ValueError("Retriever cache is inconsistent: dense_corpus length mismatch")
        
        # 如果存在嵌入向量文件，则从numpy文件加载
        embeddings_path = Path(retriever_cache_embeddings_file)
        if embeddings_path.exists():
            retriever.embeddings = np.load(embeddings_path)

        # 重新构建 BM25 索引，确保缓存加载后仍然是混合检索而非退化成纯 Dense
        if retriever.corpus:
            try:
                from rank_bm25 import BM25Okapi
                retriever.tokenized_corpus = [retriever._tokenize_for_bm25(doc) for doc in retriever.corpus]
                retriever.bm25 = BM25Okapi(retriever.tokenized_corpus)
            except Exception:
                retriever.bm25 = None
                retriever.tokenized_corpus = []
            
        return retriever
    
    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str,
                              dense_weight: float = 0.70,
                              embeddings_cache_file: str = None,
                              doc_texts: List[str] = None,
                              dense_doc_texts: List[str] = None) -> 'HybridRetriever':
        """
        从内存中的记忆对象加载检索器状态

        参数:
            memories: 记忆字典
            model_name: 使用的模型名称
            embeddings_cache_file: 可选的嵌入向量缓存文件路径，如果提供则复用已有嵌入向量
            doc_texts: 可选的文档文本列表，与 memories 顺序一致；
                       传入后 BM25 索引将与 add_documents 路径保持一致

        返回:
            创建的检索器实例
        """
        if doc_texts is not None:
            all_docs = doc_texts
        else:
            all_docs = [", ".join(m.keywords) for m in memories.values()]
        if dense_doc_texts is not None:
            all_dense_docs = dense_doc_texts
        else:
            all_dense_docs = list(all_docs)
        if len(all_docs) != len(all_dense_docs):
            raise ValueError("doc_texts and dense_doc_texts must have the same length")
        retriever = cls(model_name, dense_weight=dense_weight)
        
        # 如果提供了嵌入向量缓存文件且文件存在，尝试加载
        if embeddings_cache_file and Path(embeddings_cache_file).exists():
            try:
                cached_embeddings = np.load(embeddings_cache_file)
                # 验证嵌入向量数量是否匹配
                if len(cached_embeddings) == len(all_dense_docs):
                    retriever.corpus = all_docs
                    retriever.dense_corpus = all_dense_docs
                    retriever.embeddings = cached_embeddings
                    retriever.document_ids = {doc: idx for idx, doc in enumerate(all_docs)}
                    try:
                        from rank_bm25 import BM25Okapi
                        retriever.tokenized_corpus = [retriever._tokenize_for_bm25(doc) for doc in all_docs]
                        retriever.bm25 = BM25Okapi(retriever.tokenized_corpus)
                    except Exception:
                        retriever.bm25 = None
                        retriever.tokenized_corpus = []
                    
                    print(f"✓ 成功复用已有嵌入向量缓存")
                    return retriever
                else:
                    print(f"⚠ 嵌入向量数量不匹配（缓存: {len(cached_embeddings)}, 文档: {len(all_dense_docs)}），将重新生成")
            except Exception as e:
                print(f"⚠ 加载嵌入向量缓存失败: {e}，将重新生成")
        
        # 如果没有缓存或加载失败，使用原始方法
        retriever.add_documents(all_docs, dense_documents=all_dense_docs)
        return retriever
    
    def add_documents(
        self,
        documents: List[str],
        dense_documents: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> bool:
        """
        一次性添加文档到语义索引中 (支持批量向量化)
        
        参数:
            documents: 文档文本列表
            batch_size: 批量向量化大小
            
        返回:
            True表示成功添加
        """
        if not documents:
            return False

        if dense_documents is None:
            dense_documents = documents
        if len(documents) != len(dense_documents):
            raise ValueError("documents and dense_documents must have the same length")

        doc_pairs = [
            (doc, dense_doc)
            for doc, dense_doc in zip(documents, dense_documents)
            if doc not in self.document_ids
        ]
        if not doc_pairs:
            return False

        new_docs = [doc for doc, _ in doc_pairs]
        new_dense_docs = [dense_doc for _, dense_doc in doc_pairs]

        # 创建嵌入向量 (使用 batch_size 提速)
        new_embeddings = self.model.encode(new_dense_docs, batch_size=batch_size)
        
        # 修复：确保 new_embeddings 是 numpy array 且为二维 (N, D)
        if hasattr(new_embeddings, 'numpy'):
            new_embeddings = new_embeddings.numpy()
        if len(new_embeddings.shape) == 1:
            new_embeddings = new_embeddings.reshape(1, -1)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        doc_idx = len(self.corpus)
        for document, dense_document in zip(new_docs, new_dense_docs):
            self.corpus.append(document)
            self.dense_corpus.append(dense_document)
            self.document_ids[document] = doc_idx
            doc_idx += 1

        self._rebuild_bm25()

        return True

    def _rebuild_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            self.tokenized_corpus = [self._tokenize_for_bm25(doc) for doc in self.corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        except Exception:
            self.bm25 = None
            self.tokenized_corpus = []

    def add_document(self, document: str, dense_document: Optional[str] = None) -> bool:
        """
        向检索器添加单个文档
        
        参数:
            document: 要添加的文本内容
            
        返回:
            bool: 如果文档已添加返回True，如果文档已存在返回False
        """
        # 检查文档是否已存在
        if document in self.document_ids:
            return False
            
        if dense_document is None:
            dense_document = document

        # 添加到语料库并获取索引
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.dense_corpus.append(dense_document)
        self.document_ids[document] = doc_idx
        
        # 更新嵌入向量
        # 修复：使用 numpy 进行拼接，避免 tensor 类型不匹配错误
        # 修复：将一维数组转换为二维数组 (1, D)
        doc_embedding = self.model.encode([dense_document])
        if hasattr(doc_embedding, 'numpy'): # If it's a tensor
            doc_embedding = doc_embedding.numpy()
        
        # 确保形状是 (1, D)
        if len(doc_embedding.shape) == 1:
            doc_embedding = doc_embedding.reshape(1, -1)
            
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embedding])
            
        self._rebuild_bm25()
            
        return True
        
    def retrieve(self, query: str, k: int = 5, return_scores: bool = False) -> Union[List[int], Tuple[List[int], List[float]]]:
        """
        使用混合评分检索文档
        
        参数:
            query: 查询文本
            k: 返回的文档数量
            return_scores: 是否返回分数
            
        返回:
            相关文档的索引列表，如果return_scores=True，则返回(索引列表, 分数列表)
        """
        if not self.corpus:
            return ([] if not return_scores else ([], []))
        
        if self.embeddings is None:
            return ([] if not return_scores else ([], []))

        hybrid_scores = self.get_hybrid_scores(query)
        
        # 获取前k个索引
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        if return_scores:
            top_k_scores = hybrid_scores[top_k_indices]
            return top_k_indices.tolist(), top_k_scores.tolist()
        else:
            return top_k_indices.tolist()

    def get_scores_by_emb(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        [HyDE 支持方法] 直接接受预计算好的向量计算所有文档的余弦相似度分数。
        由 find_related_memories_advanced 调用，传入 HyDE 生成的假设性回复的向量。

        参数:
            query_embedding: shape (D,) 的 numpy 向量

        返回:
            shape (N,) 的分数数组，每个元素对应语料库中的一个文档
        """
        if self.embeddings is None or len(self.corpus) == 0:
            return np.array([])

        # 防御性转换：确保是 NumPy 数组
        if hasattr(query_embedding, 'numpy'):
            query_embedding = query_embedding.numpy()
            
        # 强制 query_embedding 变为一维以进行点积计算
        query_embedding = np.squeeze(query_embedding)
        
        embeddings_arr = np.array(self.embeddings)
        norm_q = np.linalg.norm(query_embedding)
        norm_e = np.linalg.norm(embeddings_arr, axis=1)
        norm_e[norm_e == 0] = 1e-10
        if norm_q == 0:
            norm_q = 1e-10

        return np.dot(embeddings_arr, query_embedding) / (norm_e * norm_q)

    def get_dense_scores_normalized(self, query_embedding: np.ndarray) -> np.ndarray:
        """返回 min-max 归一化的 Dense cosine 相似度分数。"""
        raw = self.get_scores_by_emb(query_embedding)
        if len(raw) == 0:
            return np.array([])
        return self._min_max_norm_array(raw)

    def get_bm25_scores_normalized(self, query: str) -> np.ndarray:
        """返回 min-max 归一化的 BM25 分数；BM25 不可用时返回全零。"""
        if self.bm25 is None or not self.tokenized_corpus:
            return np.zeros(len(self.corpus), dtype=float)
        tokens = self._tokenize_for_bm25(query)
        raw = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        return self._min_max_norm_array(raw)


class AgenticDecomposer:
    """
    代理式查询分解器 (Agentic Query Decomposer)
    用于将复杂的多意图查询分解为多个原子级子查询。
    附带核心意图(core intent)以防止分解出来的子问题产生语义偏离(retrieval drift)。
    """
    def __init__(self, llm_controller: LLMController):
        self.llm_controller = llm_controller
        self.system_prompt = '''You are an expert AI query decomposer.
Your task is to break down complex user queries into smaller, independent atomic sub-queries that are easier to search in a vector database.
You must also identify the "core_intent" of the original query to ensure no sub-query drifts away from the main topic.

Output your response ONLY as a JSON object with this structure:
{
    "core_intent": "The main goal or topic",
    "sub_queries": ["sub query 1", "sub query 2"]
}
'''

    def decompose(self, query: str) -> dict:
        """
        分解问题为核心意图和独立的子问题。
        """
        try:
            prompt = f"Decompose the following complex query:\n{query}"
            
            response_format = {
                "type": "json_schema", 
                "json_schema": {
                    "name": "decomposition",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "core_intent": {"type": "string"},
                            "sub_queries": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["core_intent", "sub_queries"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
            
            response = self.llm_controller.llm.get_completion(
                self.system_prompt + "\n" + prompt, 
                response_format=response_format,
                temperature=0.3
            )
            
            # 基础 JSON 清理
            clean_resp = response.strip()
            if not clean_resp.startswith("{"):
                clean_resp = clean_resp[clean_resp.find("{"):]
            if not clean_resp.endswith("}"):
                clean_resp = clean_resp[:clean_resp.rfind("}")+1]
                
            return json.loads(clean_resp)
        except Exception as e:
            print(f"⚠ Decomposition failed: {e}. Falling back to original query.")
            return {"core_intent": query, "sub_queries": [query]}

class ReflectionVerifier:
    """
    反思验证器 (Reflection Verifier)
    用于在检索出片段后，使用反思令牌(<reflection>)验证片段的相关性；
    并在最后验证所有搜集的证据是否构成针对核心意图的逻辑闭环。
    """
    def __init__(self, llm_controller: LLMController):
        self.llm_controller = llm_controller

    def verify_relevance(self, query: str, document: str) -> dict:
        """
        验证单个文档片段是否能够回答给定的问题。
        引入 reflection 强制模型进行思考打分。
        """
        prompt = f"""Evaluate if the following document fragment is relevant to the search query.
Query: {query}
Document: {document}

Before giving your final judgment, you MUST write your thought process inside a <reflection> tag.
Then provide a boolean "is_relevant" score.

Output as JSON:
{{
    "reflection": "<reflection> your thought process here </reflection>",
    "is_relevant": true or false
}}
"""
        response_format = {
            "type": "json_schema", 
            "json_schema": {
                "name": "relevance_check",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reflection": {"type": "string"},
                        "is_relevant": {"type": "boolean"}
                    },
                    "required": ["reflection", "is_relevant"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format=response_format, temperature=0.1)
            
            clean_resp = response.strip()
            if not clean_resp.startswith("{"):
                clean_resp = clean_resp[clean_resp.find("{"):]
            if not clean_resp.endswith("}"):
                clean_resp = clean_resp[:clean_resp.rfind("}")+1]
                
            return json.loads(clean_resp)
        except Exception as e:
            print(f"⚠ Relevance reflection failed: {e}. Assuming true.")
            return {"reflection": "<reflection> Error </reflection>", "is_relevant": True}

    def verify_logical_closure(self, query: str, evidences: list) -> dict:
        """
        验证搜集到的所有证据是否能够针对核心意图形成逻辑闭环。
        如果逻辑不完整，返回缺少的关键信息提示。
        """
        evidences_str = "\n".join([f"[{i+1}] {e}" for i, e in enumerate(evidences)])
        prompt = f"""Evaluate if the following collected evidences form a complete logical closed loop to fully answer the main query.
Query: {query}
Evidences:
{evidences_str}

Consider: Is there any critical missing piece of information needed to form a complete answer?
Write your thought inside <reflection> tag, and set "is_closed_loop" to true if sufficient, false otherwise.
If false, populate "missing_information" with exactly what is missing.

Output as JSON:
{{
    "reflection": "<reflection> thought </reflection>",
    "is_closed_loop": true or false,
    "missing_information": "what is missing, or empty string if true"
}}
"""
        response_format = {
            "type": "json_schema", 
            "json_schema": {
                "name": "closure_check",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reflection": {"type": "string"},
                        "is_closed_loop": {"type": "boolean"},
                        "missing_information": {"type": "string"}
                    },
                    "required": ["reflection", "is_closed_loop", "missing_information"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format=response_format, temperature=0.2)
            clean_resp = response.strip()
            if not clean_resp.startswith("{"):
                clean_resp = clean_resp[clean_resp.find("{"):]
            if not clean_resp.endswith("}"):
                clean_resp = clean_resp[:clean_resp.rfind("}")+1]
            return json.loads(clean_resp)
        except Exception as e:
            print(f"⚠ Logical closure check failed: {e}. Assuming closed loop.")
            return {"reflection": "<reflection> Error </reflection>", "is_closed_loop": True, "missing_information": ""}

class AgenticMemorySystem:
    """
    智能记忆管理系统
    
    基于语义检索的记忆管理系统
    """
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3',
                 reranker_model_name: str = 'BAAI/bge-reranker-v2-minicpm-layerwise',
                 reranker_cutoff_layer: int = 28,
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 use_reranking: bool = True):
        """
        初始化智能记忆系统
        
        参数:
            model_name: 嵌入模型名称
            llm_backend: LLM后端类型（"openai", "ollama", "sglang"）
            llm_model: LLM模型名称
            evo_threshold: 记忆演化阈值，每达到此数量的演化会触发记忆整合
            api_key: API密钥（可选）
            api_base: API基础URL（可选）
            sglang_host: SGLang服务器主机（可选）
            sglang_port: SGLang服务器端口（可选）
            use_reranking: 是否启用重排序（使用CrossEncoder提升检索精度）
        """
        self.memories = {}  # id -> MemoryNote
        # Dense 为主、BM25 为辅。对对话记忆闭集检索，0.70/0.30 通常比纯 Dense 更稳。
        self.retriever_alpha = 0.70
        self.retriever = HybridRetriever(model_name, dense_weight=self.retriever_alpha)
        self.llm_controller = LLMController(
            backend=llm_backend,
            model=llm_model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        # 重排序相关
        self.use_reranking = use_reranking
        self.reranker_model_name = reranker_model_name
        self.reranker_cutoff_layer = int(reranker_cutoff_layer)
        self.reranker = None  # 延迟加载
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {{context}}
                                content: {{content}}
                                keywords: {{keywords}}

                                The nearest neighbors memories:
                                {{nearest_neighbors_memories}}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? You must also output the specific type of the connection (e.g. "causes", "results_in", "happens_before", "happens_after", "similar_to") in `connection_types`, and give the updated tags of this memory.
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {{neighbor_number}}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "connection_types": ["connection type strings, corresponding to each connection id"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]]
                                }}
                                '''
        self.evo_cnt = 0  # 演化计数器
        self.evo_threshold = evo_threshold  # 演化阈值

        # ── GraphRAG 社区聚类（新增）──────────────────────────────────
        # 图边列表：{src_id, tgt_id, weight, type}
        self.graph_edges: List[dict] = []
        # 社区摘要字典：community_id -> CommunitySummary
        self.communities: Dict[str, CommunitySummary] = {}
        # 社区 embedding 矩阵（与 community_ids_list 行对齐）
        self.community_embeddings: Optional[np.ndarray] = None
        self.community_ids_list: List[str] = []
        # 触发全量重聚类的 note 新增数阈值
        self.community_rebuild_interval: int = 50
        # 已添加 note 计数（用于触发聚类）
        self.note_total_count: int = 0
        # 语义边阈值（cosine similarity）
        # 0.70 过低导致边数爆炸 → Louvain 产生少量巨型社区 → 社区过滤失效
        # 提高至 0.82 可减少约 60% 冗余边，社区粒度更细
        self.edge_semantic_threshold: float = 0.82
        # 关键词共现边阈值（Jaccard）
        self.edge_jaccard_threshold: float = 0.20
        # 图状态缓存脏标记
        self._graph_is_dirty: bool = False
        # ────────────────────────────────────────────────────────────

    def _get_note_by_doc_idx(self, idx: int) -> Optional[MemoryNote]:
        """
        通过语料库索引获取对应的 MemoryNote
        修复了由于检索器过滤重复文档导致索引与 self.memories 错位的问题
        """
        if idx >= len(self.retriever.corpus):
            all_memories = list(self.memories.values())
            return all_memories[idx] if idx < len(all_memories) else None
            
        doc_text = self.retriever.corpus[idx]
        
        # 1. 尝试从新格式 "ID:<uuid> CONTENT:..." 中提取 ID
        if doc_text.startswith("ID:"):
            end_id = doc_text.find(" CONTENT:")
            if end_id != -1:
                note_id = doc_text[3:end_id]
                if note_id in self.memories:
                    return self.memories[note_id]
                    
        # 2. 回退：兼容旧的缓存格式
        for note in self.memories.values():
            if doc_text == "content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags):
                return note
            metadata_text = f"{note.context} {' '.join(note.keywords)} {' '.join(note.tags)}"
            if doc_text == note.content + " , " + metadata_text:
                return note
                
        # 3. 最终回退：直接位置映射（可能错位，但作为兜底）
        all_memories = list(self.memories.values())
        if idx < len(all_memories):
            return all_memories[idx]
        return None

    def _generate_retrieval_text(self, note: MemoryNote) -> str:
        """生成用于 BM25 / 图检索的富文本视图。"""
        date_str = ""
        if note.timestamp and len(note.timestamp) >= 8:
            try:
                from datetime import datetime as _dt
                dt = _dt.strptime(note.timestamp[:8], "%Y%m%d")
                date_str = dt.strftime("%B %d %Y")  # e.g. "May 07 2023"
            except ValueError:
                date_str = note.timestamp[:8]
        date_field = f" DATE:{date_str}" if date_str else ""
        return f"ID:{note.id}{date_field} CONTENT:{note.content} CONTEXT:{note.context} KEYWORDS:{', '.join(note.keywords)} TAGS:{', '.join(note.tags)}"

    def _generate_dense_text(self, note: MemoryNote) -> str:
        """生成用于 dense embedding 的纯语义视图。"""
        return note.content

    def _generate_doc_text(self, note: MemoryNote) -> str:
        """兼容旧调用，默认返回检索富文本视图。"""
        return self._generate_retrieval_text(note)

    def add_notes_batch(self, contents: List[str], times: List[str] = None, **kwargs) -> List[str]:
        """
        批量添加新的记忆笔记并进行批量向量化 (Batch Embedding)
        
        参数:
            contents: 记忆内容文本列表
            times: 时间戳列表（可选）
            **kwargs: 其他可选参数
            
        返回:
            新添加记忆的ID列表
        """
        if not contents:
            return []
            
        if times is None:
            times = [None] * len(contents)
            
        note_ids = []
        retrieval_docs_to_add = []
        dense_docs_to_add = []
        new_notes: List[MemoryNote] = []
        existing_notes_before_batch = list(self.memories.values())
        
        for content, time in zip(contents, times):
            note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
            evo_label, note = self.process_memory(note)
            self.memories[note.id] = note
            
            retrieval_docs_to_add.append(self._generate_retrieval_text(note))
            dense_docs_to_add.append(self._generate_dense_text(note))
            
            note_ids.append(note.id)
            new_notes.append(note)
            
            if evo_label == True:
                self.evo_cnt += 1
                
        # 批量进行 Embedding 并更新检索器
        if retrieval_docs_to_add:
            self.retriever.add_documents(
                retrieval_docs_to_add,
                dense_documents=dense_docs_to_add,
                batch_size=32,
            )
        edge_context_notes = list(existing_notes_before_batch)

        # Batch 路径必须在 retriever 更新完成后再建图，
        # 否则 semantic / BM25 边会基于错误或缺失的索引构建。
        for note in new_notes:
            self._build_edges_for_new_note(note, existing_notes=edge_context_notes)
            edge_context_notes.append(note)
            self.note_total_count += 1
            self._graph_is_dirty = True
            
        if self.evo_cnt > 0 and self.evo_cnt % self.evo_threshold == 0:
            self.consolidate_memories()
            
        if self.note_total_count % self.community_rebuild_interval == 0:
            self.rebuild_communities()
            
        return note_ids

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """
        添加新的记忆笔记
        
        参数:
            content: 记忆内容文本
            time: 时间戳（可选）
            **kwargs: 其他可选参数
            
        返回:
            新添加记忆的ID
        """
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        
        # 更新检索器的文档列表
        # all_docs = [m.content for m in self.memories.values()]
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        retrieval_text = self._generate_retrieval_text(note)
        dense_text = self._generate_dense_text(note)
        self.retriever.add_document(retrieval_text, dense_document=dense_text)
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        # ── GraphRAG：构建图边 + 触发社区重聚类 ─────────────────────
        self._build_edges_for_new_note(note)
        self.note_total_count += 1
        
        # 标记图为 dirty，下次需要社区时再聚类
        self._graph_is_dirty = True
        
        if self.note_total_count % self.community_rebuild_interval == 0:
            self.rebuild_communities()
        # ────────────────────────────────────────────────────────────

        return note.id
    
    def consolidate_memories(self):
        """
        整合记忆：使用新文档更新检索器
        
        此函数重新初始化检索器并使用所有记忆文档更新它，
        包括它们的上下文、关键词和标签，以确保检索系统具有所有记忆的最新状态。
        """
        # 使用相同的模型重置检索器
        try:
            # 尝试通过get_config_dict获取模型名称（如果可用）
            model_name = getattr(self.retriever, 'model_name', 'BAAI/bge-m3')
        except (AttributeError, KeyError):
            # 回退：使用类初始化时的模型名称
            model_name = 'BAAI/bge-m3'
        
        self.retriever = HybridRetriever(model_name, dense_weight=self.retriever_alpha)
        
        # 收集所有记忆文档并一次性添加
        all_docs = []
        all_dense_docs = []
        for memory in self.memories.values():
            # 将记忆元数据合并为单个可搜索文档
            all_docs.append(self._generate_retrieval_text(memory))
            all_dense_docs.append(self._generate_dense_text(memory))
        
        # 一次性添加所有文档（HybridRetriever的add_documents会重新初始化索引）
        if all_docs:
            self.retriever.add_documents(all_docs, dense_documents=all_dense_docs)

        self._rebuild_graph_edges()

        # ── GraphRAG：同步重建社区 ──────────────────────────────────────
        if len(self.memories) >= 5:
            self.rebuild_communities()
        # ────────────────────────────────────────────────────────────────
    
    def _rebuild_graph_edges(self) -> None:
        """基于当前 memories + retriever 状态全量重建图边。"""
        self.graph_edges = []
        ordered_notes = list(self.memories.values())
        for note in ordered_notes:
            note.id_based_links = []
            note.community_id = None

        existing_notes: List[MemoryNote] = []
        for note in ordered_notes:
            self._build_edges_for_new_note(note, existing_notes=existing_notes)
            existing_notes.append(note)

        self._graph_is_dirty = True

    def process_memory(self, note: MemoryNote) -> bool:
        """
        处理记忆笔记并返回演化标签
        
        参数:
            note: 要处理的记忆笔记对象
            
        返回:
            (should_evolve, note) 元组，should_evolve表示是否需要演化
        """
        # 使用内部方法获取索引，避免解包错误
        indices = self._retrieve_indices(note.content, k=5)
        
        # 构建邻居记忆字符串
        neighbor_memory = ""
        for i in indices:
            mem = self._get_note_by_doc_idx(i)
            if mem:
                neighbor_memory += "memory index:" + str(i) + "\t talk start time:" + mem.timestamp + "\t memory content: " + mem.content + "\t memory context: " + mem.context + "\t memory keywords: " + str(mem.keywords) + "\t memory tags: " + str(mem.tags) + "\n"

        # 使用安全的字符串替换，避免大括号注入导致 format 崩溃
        prompt_memory = self.evolution_system_prompt.replace("{{context}}", note.context)\
                                                    .replace("{{content}}", note.content)\
                                                    .replace("{{keywords}}", str(note.keywords))\
                                                    .replace("{{nearest_neighbors_memories}}", neighbor_memory)\
                                                    .replace("{{neighbor_number}}", str(len(indices)))
        print("prompt_memory", prompt_memory)
        response = self.llm_controller.llm.get_completion(
            prompt_memory,response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean",
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    }
                                },
                                "connection_types": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve","actions","suggested_connections","connection_types","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
        )
        try:
            print("response", response, type(response))
            # 清理响应，以防有多余的文本
            response_cleaned = response.strip()
            # 尝试从其他文本中提取JSON内容
            if not response_cleaned.startswith('{'):
                start_idx = response_cleaned.find('{')
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith('}'):
                end_idx = response_cleaned.rfind('}')
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx+1]
            
            response_json = json.loads(response_cleaned)
            print("response_json", response_json, type(response_json))
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response}")
            # 解析失败时返回默认值
            return False, note
        should_evolve = response_json["should_evolve"]
        if should_evolve:
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json.get("suggested_connections", [])
                    connection_types = response_json.get("connection_types", [])
                    new_tags = response_json.get("tags_to_update", [])
                    
                    # 修复：将返回的邻居位置索引(int)转换为实际的记忆ID(str)，消除links字段的类型混淆
                    # 同时将推断出的结构化连边关系保存下来
                    for list_idx, idx in enumerate(suggest_connections):
                        if isinstance(idx, int):
                            target_note = self._get_note_by_doc_idx(idx)
                            if target_note:
                                rel_type = connection_types[list_idx] if list_idx < len(connection_types) else "similar_to"
                                # 以前简单的存储链接ID，现在我们存成字典格式 {"id": id, "type": type}
                                # 为了兼容原代码的 id 列表格式，我们仍然往 note.links 里加 string 
                                # 但在 note_id_based_links 之外，我们要提供一种方式把它暴露出去
                                # 这里我们临时在 note.links 存 dict，如果别的地方不兼容再改。
                                # 由于 _build_edges_for_new_note 是处理它的下游，我们可以把它传过去
                                if target_note.id not in [link.get("id") if isinstance(link, dict) else link for link in note.links]:
                                    note.links.append({"id": target_note.id, "type": rel_type})
                                
                    note.tags = new_tags or note.tags
                    note.normalize_metadata()
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    # 如果小语言模型输出的数量少于邻居数量，则按顺序使用新标签和上下文
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        mem_idx = indices[i]
                        notetmp = self._get_note_by_doc_idx(mem_idx)
                        if not notetmp:
                            continue
                            
                        # 查找某个记忆
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = notetmp.context
                            
                        # 向记忆添加标签
                        notetmp.tags = tag or notetmp.tags
                        notetmp.context = context
                        notetmp.normalize_metadata()
                        self.memories[notetmp.id] = notetmp
        note.normalize_metadata()
        return should_evolve,note

    def _get_reranker(self):
        """
        延迟加载重排序器

        返回:
            具有 predict(pairs) 接口的重排序器实例，如果不可用则返回 None
        """
        if self.reranker is None and self.use_reranking:
            try:
                self.reranker = load_reranker(
                    self.reranker_model_name,
                    cutoff_layer=self.reranker_cutoff_layer,
                )
                print(f"Loaded reranker: {self.reranker_model_name}")
            except Exception as e:
                print(f"Warning: Failed to load reranker {self.reranker_model_name}: {e}, reranking disabled")
                self.use_reranking = False
        return self.reranker

    def rerank_memories(self, query: str, retrieved_indices: List[int], k: int = 5) -> List[int]:
        """
        对检索结果进行重排序，使用CrossEncoder提升精度
        
        参数:
            query: 查询文本
            retrieved_indices: 第一轮检索得到的索引列表
            k: 最终返回的数量
        
        返回:
            重排序后的索引列表
        """
        if not self.use_reranking or len(retrieved_indices) <= k:
            return retrieved_indices[:k]
        
        reranker = self._get_reranker()
        if reranker is None:
            return retrieved_indices[:k]
        
        # 构建查询-文档对
        pairs = []
        valid_indices = []
        for idx in retrieved_indices:
            mem = self._get_note_by_doc_idx(idx)
            if not mem:
                continue
            # 构建文档文本（用于重排序）
            # 优化顺序：关键词 + 标签 + 内容 + 上下文（确保关键元数据不被截断）
            doc_text = f"{', '.join(mem.keywords)} , {', '.join(mem.tags)} , {mem.content} , {mem.context}"
            pairs.append([query, doc_text])
            valid_indices.append(idx)
        
        # 使用交叉编码器评分（更精确但更慢）
        try:
            scores = reranker.predict(pairs)
            # 按分数排序（分数越高越相关）
            ranked_indices = [valid_indices[i] for i in np.argsort(scores)[::-1]]
            return ranked_indices[:k]
        except Exception as e:
            print(f"⚠ Warning: Reranking failed: {e}, using original order")
            return retrieved_indices[:k]

    def _retrieve_indices(self, query: str, k: int = 5) -> List[int]:
        """
        内部辅助方法：获取相关记忆的索引
        """
        if not self.memories:
            return []
            
        initial_indices, initial_scores = self.retriever.retrieve(query, k, return_scores=True)
        return initial_indices

    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """
        使用混合检索查找相关记忆
        返回记忆对象列表
        """
        indices = self._retrieve_indices(query, k)
        return [self._get_note_by_doc_idx(i) for i in indices if self._get_note_by_doc_idx(i)]



    def agentic_retrieve(self, query: str, k: int = 5, max_verify_per_subquery: int = 8,
                         include_community_context: bool = True,
                         dense_weight: Optional[float] = None) -> Union[str, Dict[str, Any]]:
        """
        基于 Agentic Decomposition 和 Reflection 的智能检索。
        将复杂问题分治后独立检索并过滤。
        """
        # 1. 初始化组件
        decomposer = AgenticDecomposer(self.llm_controller)
        verifier = ReflectionVerifier(self.llm_controller)
        
        # 2. 任务分解
        decomp_result = decomposer.decompose(query)
        sub_queries = decomp_result.get("sub_queries", [query])
        core_intent = decomp_result.get("core_intent", query)
        
        all_relevant_contexts = []
        seen_documents = set()
        involved_communities = set()
        
        # 3. 定向检索与反思过滤
        for sub_q in sub_queries:
            # 加上核心意图，防止偏离
            context_query = f"Intent: {core_intent} Sub-query: {sub_q}"
            
            # 使用升级版的超能漫游检索
            advanced_result = self.find_related_memories_advanced(
                context_query,
                k=k,
                include_community_context=include_community_context,
                dense_weight=dense_weight,
            )
            raw_memories = advanced_result.get("notes", [])
            
            if include_community_context:
                comms = advanced_result.get("communities", [])
                for c in comms:
                    involved_communities.add(c.community_id)
                    
            verify_pool = raw_memories[:max_verify_per_subquery]
            
            for mem in verify_pool:
                doc_text = f"Content: {mem.content} Keywords: {mem.keywords}"
                if doc_text in seen_documents:
                    continue
                seen_documents.add(doc_text)
                
                # 4. 反思节点相关性
                rel_check = verifier.verify_relevance(sub_q, doc_text)
                if rel_check.get("is_relevant", True):
                    # 如果相关，则计入最终证据
                    all_relevant_contexts.append(
                        f"[Verified Evidence for '{sub_q}'] {doc_text}\n"
                        f"Reflection: {rel_check.get('reflection', '')}"
                    )
        
        # 如果全部被过滤掉（非常极端），回退到基础检索
        if not all_relevant_contexts:
            print("⚠ Reflection filtered out all results. Falling back to simple retrieval.")
            fallback_memories = self.find_related_memories(query, k=k)
            for mem in fallback_memories:
                all_relevant_contexts.append(f"[Fallback Evidence] {mem.content} Keywords: {mem.keywords}")
                if mem.community_id:
                    involved_communities.add(mem.community_id)
            
        # 5. 验证假设文档的逻辑闭环
        closure_check = verifier.verify_logical_closure(query, all_relevant_contexts)
        
        # 如果逻辑非闭环，尝试用缺失信息进行二次补充检索 (Anti-drift Follow-up)
        if not closure_check.get("is_closed_loop", True):
            missing_info = closure_check.get("missing_information", "")
            print(f"↻ Logical loop incomplete. Missing: {missing_info}. Triggering follow-up retrieval...")
            followup_query = f"Intent: {core_intent} Missing: {missing_info}"
            followup_result = self.find_related_memories_advanced(
                followup_query,
                k=2,
                include_community_context=include_community_context,
                dense_weight=dense_weight,
            )
            followup_memories = followup_result.get("notes", [])
            
            if include_community_context:
                comms = followup_result.get("communities", [])
                for c in comms:
                    involved_communities.add(c.community_id)
            
            for mem in followup_memories:
                doc_text = f"Content: {mem.content} Keywords: {mem.keywords}"
                if doc_text not in seen_documents:
                    seen_documents.add(doc_text)
                    all_relevant_contexts.append(
                        f"[Follow-up Evidence for '{missing_info}'] {doc_text}\n"
                        f"Reflection: <reflection> Retrieved to resolve logical gap </reflection>"
                    )
            
        final_context = "\n---\n".join(all_relevant_contexts)
        
        if include_community_context and involved_communities:
            community_context = ""
            for cid in involved_communities:
                if cid in self.communities:
                    cs = self.communities[cid]
                    kws_str = ", ".join(cs.keywords[:5]) if cs.keywords else "None"
                    community_context += f"[{cs.title}] {cs.summary} (Keywords: {kws_str})\n"
            
            # 返回字典以匹配 _format_retrieval_context 的处理逻辑
            return {
                "notes": [], # _format_retrieval_context 期望的 notes 列表，但由于我们已经自己构建了文本，可以返回空列表，然后直接将 final_context 作为 community_context 的前缀
                "community_context": f"{final_context}\n\n{community_context}"
            }
            
        return final_context

    # ══════════════════════════════════════════════════════════════════════
    # ██  GraphRAG 社区聚类方法（新增）
    # ══════════════════════════════════════════════════════════════════════

    def _build_edges_for_new_note(
        self,
        note: MemoryNote,
        existing_notes: Optional[List[MemoryNote]] = None,
    ) -> None:
        """
        为新加入的 note 自动构建三类图边：
          1. semantic  : 语义 cosine 相似度 ≥ edge_semantic_threshold
          2. temporal  : timestamp 前缀相同（同一 session）且连续
          3. entity_shared : 关键词 Jaccard 相似度 ≥ edge_jaccard_threshold

        构建完成后，同时更新 note.id_based_links（双向）。
        """
        if not self.memories:
            return

        if self.retriever.embeddings is None or len(self.retriever.embeddings) == 0:
            return
        new_doc_key = self._generate_doc_text(note)

        # 始终通过文档 key 回查 embedding 行，避免 batch add_documents
        # 时“最后一条 embedding 就是当前 note”这一假设失效。
        new_idx = self.retriever.document_ids.get(new_doc_key)
        if new_idx is None or new_idx >= len(self.retriever.embeddings):
            return

        new_emb = self.retriever.embeddings[new_idx].reshape(1, -1)   # shape (1, D)

        if existing_notes is None:
            existing_notes = list(self.memories.values())
        # 不包含刚加入的那条（self.memories 在此之前已写入）
        # 但 note 此时已在 self.memories 中，所以要排除自身
        existing_notes = [n for n in existing_notes if n.id != note.id]

        if not existing_notes:
            return

        # ── 1. 语义边 ──────────────────────────────────────────────────
        # 从检索器的 document_ids 映射中获取各 note 的 embedding 行索引
        # 这样即使 consolidate_memories() 重建了检索器，对齐关系也始终正确
        if self.retriever.embeddings is not None and self.retriever.embeddings.shape[0] > 1:
            for prev_note in existing_notes:
                # 查找该 note 对应的检索器文档 key（与 add_note 写入格式一致）
                prev_doc_key = self._generate_doc_text(prev_note)
                prev_idx = self.retriever.document_ids.get(prev_doc_key)
                if prev_idx is None:
                    continue   # 该 note 在检索器中找不到，跳过
                prev_emb = self.retriever.embeddings[prev_idx].reshape(1, -1)
                sim = float(cosine_similarity(new_emb, prev_emb)[0][0])
                if sim >= self.edge_semantic_threshold:
                    self.graph_edges.append({
                        "src_id": note.id,
                        "tgt_id": prev_note.id,
                        "weight": round(sim, 4),
                        "type": "semantic"
                    })
                    if prev_note.id not in note.id_based_links:
                        note.id_based_links.append(prev_note.id)
                    if note.id not in prev_note.id_based_links:
                        prev_note.id_based_links.append(note.id)

        # ── 2. 改进的时序边（实体生命周期跨会话追踪） ──────────────────────────
        # 不再仅仅局限于“同一天最近5条”，而是基于共享实体（Entity/Keywords）追踪历史时序
        # 如果两条记忆包含相同的主体，且时间有先后，则建立跨越时间的长程关联，强化时间线的纵向追踪
        new_kw_set = set(k.lower() for k in note.keywords)
        has_temporal_link = False
        
        if new_kw_set:
            # 找到历史中含有相同实体的最近几条记忆
            shared_entity_notes = []
            for prev_note in reversed(existing_notes):
                prev_kw_set = set(k.lower() for k in prev_note.keywords)
                if prev_kw_set and len(new_kw_set & prev_kw_set) > 0:
                    shared_entity_notes.append(prev_note)
                if len(shared_entity_notes) >= 3: # 追踪同一个实体的最近3个生命周期快照
                    break
                    
            for prev_note in shared_entity_notes:
                # 只在有时间戳时比较，或默认新插入的在后
                self.graph_edges.append({
                    "src_id": prev_note.id, # 过去指向现在
                    "tgt_id": note.id,
                    "weight": 0.7,
                    "type": "temporal_entity_evolution"
                })
                if note.id not in prev_note.id_based_links:
                    prev_note.id_based_links.append(note.id)
                if prev_note.id not in note.id_based_links:
                    note.id_based_links.append(prev_note.id)
                has_temporal_link = True
                
        # 兜底：如果没有任何实体交集，且时间非常近，依然保持基础时序连通性
        if not has_temporal_link:
            new_session = note.timestamp[:8] if note.timestamp else ""
            for prev_note in existing_notes[-2:]:   # 缩小纯瞎连的范围到最近2条
                prev_session = prev_note.timestamp[:8] if prev_note.timestamp else ""
                if new_session and prev_session and new_session == prev_session:
                    self.graph_edges.append({
                        "src_id": prev_note.id,
                        "tgt_id": note.id,
                        "weight": 0.5,
                        "type": "temporal_adjacent"
                    })
                    if note.id not in prev_note.id_based_links:
                        prev_note.id_based_links.append(note.id)
                    if prev_note.id not in note.id_based_links:
                        note.id_based_links.append(prev_note.id)

        # ── 3. 关键词共现边（Jaccard）──────────────────────────────────
        new_kw_set = set(k.lower() for k in note.keywords)
        if new_kw_set:
            for prev_note in existing_notes:
                prev_kw_set = set(k.lower() for k in prev_note.keywords)
                if not prev_kw_set:
                    continue
                intersection = len(new_kw_set & prev_kw_set)
                union = len(new_kw_set | prev_kw_set)
                jaccard = intersection / union if union > 0 else 0.0
                if jaccard >= self.edge_jaccard_threshold:
                    self.graph_edges.append({
                        "src_id": note.id,
                        "tgt_id": prev_note.id,
                        "weight": round(jaccard, 4),
                        "type": "entity_shared"
                    })
                    if prev_note.id not in note.id_based_links:
                        note.id_based_links.append(prev_note.id)
                    if note.id not in prev_note.id_based_links:
                        prev_note.id_based_links.append(note.id)

        # ── 5. 实体节点（虚拟节点）连边 ──────────────────────────────
        # 把每个 keyword 显式当作一个图节点（前缀 "ent:"），建立 note -> entity 的连边
        for kw in note.keywords:
            if not kw or not str(kw).strip():
                continue
            ent_id = f"ent:{str(kw).strip().lower()}"
            # note 包含 entity，权重设置为固定值（比如 1.0 代表明确包含）
            self.graph_edges.append({
                "src_id": note.id,
                "tgt_id": ent_id,
                "weight": 1.0,
                "type": "contains_entity"
            })
            # 这里我们不把 ent_id 放入 id_based_links，因为那个通常用于维护真实的 memory 节点引用
            # 但是它会进入 graph_edges，参与 PPR 和聚类
        # process_memory 已经将 note.links 写入，现在它可能包含 dict 或 string
        for link_item in note.links:
            if isinstance(link_item, dict):
                linked_id = link_item.get("id")
                rel_type = link_item.get("type", "llm_inferred")
            else:
                linked_id = link_item
                rel_type = "llm_inferred"
                
            if isinstance(linked_id, str) and linked_id in self.memories:
                if linked_id != note.id and linked_id not in note.id_based_links:
                    note.id_based_links.append(linked_id)
                    # 添加有向边（如果需要反向也可以加，但保留因果方向更有价值）
                    self.graph_edges.append({
                        "src_id": note.id,
                        "tgt_id": linked_id,
                        "weight": 0.8,
                        "type": rel_type
                    })

        # ── 6. 稀疏词汇连边 (BM25 Lexical Shared) ──────────────────────────
        # 使用 BM25 捕捉 Dense 模型可能漏掉的罕见专有名词和缩写
        if self.retriever.bm25 is not None and len(self.retriever.tokenized_corpus) > 1:
            try:
                import jieba
                doc_text = self._generate_doc_text(note)
                tokens = list(jieba.cut(doc_text)) if 'jieba' in sys.modules else doc_text.lower().split()
                # 计算新节点到整个语料库的 BM25 分数
                bm25_scores = self.retriever.bm25.get_scores(tokens)
                
                # 找出分数异常高的节点（比如超过阈值，或显著高于均值）
                # 这里我们设定一个绝对的稀疏阈值，比如 10.0，说明有罕见词重叠
                for i, score in enumerate(bm25_scores):
                    if score > 12.0: # BM25 的绝对分数没有固定上限，但 > 12 通常意味着有很强的关键词重叠
                        prev_note = self._get_note_by_doc_idx(i)
                        if prev_note and prev_note.id != note.id:
                            self.graph_edges.append({
                                "src_id": note.id,
                                "tgt_id": prev_note.id,
                                "weight": 0.9, # 强边
                                "type": "lexical_shared"
                            })
                            if prev_note.id not in note.id_based_links:
                                note.id_based_links.append(prev_note.id)
                            if note.id not in prev_note.id_based_links:
                                prev_note.id_based_links.append(note.id)
            except Exception as e:
                pass
                
    def _build_networkx_graph(self):
        """
        将 self.graph_edges 转成 networkx 无向加权图，供社区检测使用。
        在边权聚合时，增强因果和实体共现的聚类亲和力。

        返回:
            nx.Graph 对象
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx 未安装，请运行: pip install networkx")

        G = nx.Graph()
        # 添加所有 note 节点
        for note_id in self.memories:
            G.add_node(note_id)

        # 添加边（同向去重，累加不同维度的亲和力而不是单纯取最大）
        edge_map: Dict[Tuple[str, str], float] = {}
        for e in self.graph_edges:
            src, tgt, w = e["src_id"], e["tgt_id"], e["weight"]
            rel_type = e.get("type", "semantic")
            
            # 这里需要兼容 entity 节点：只有以 "ent:" 开头或者是有效 memory 时才添加
            src_valid = src.startswith("ent:") or src in self.memories
            tgt_valid = tgt.startswith("ent:") or tgt in self.memories
            
            if not src_valid or not tgt_valid:
                continue
                
            # 添加节点（如果是 entity，之前没加过的话在这里顺带加上）
            if src.startswith("ent:"): G.add_node(src)
            if tgt.startswith("ent:"): G.add_node(tgt)
                
            # 为聚集赋予不同权重：强逻辑关系更容易被划分到同一个社区
            type_clustering_weight = {
                "causes": 1.5,
                "results_in": 1.5,
                "temporal_entity_evolution": 1.2,
                "entity_shared": 1.0,
                "lexical_shared": 1.0, # 补充 BM25 连边的聚类权重
                "contains_entity": 0.8, # 让共享实体的 note 通过 entity 节点聚集
                "semantic": 0.8,
                "temporal_adjacent": 0.3 # 弱关联
            }
            
            w_adj = w * type_clustering_weight.get(rel_type, 1.0)
            
            key = (min(src, tgt), max(src, tgt))
            # 修改：同节点对的多种关系应该叠加以加强亲和力，而非单纯取最大
            edge_map[key] = min(edge_map.get(key, 0.0) + w_adj, 2.0) # 封顶2.0防极端

        for (src, tgt), w in edge_map.items():
            if w > 0:
                G.add_edge(src, tgt, weight=w)

        return G

    def _run_community_detection(self, G) -> Dict[str, int]:
        """
        在图 G 上运行 Leiden 社区检测（Louvain 的后继算法）。
        Leiden 保证社区内部严格连通，且支持 seed 参数确保结果可复现。

        参数:
            G: networkx.Graph 对象

        返回:
            {note_id: community_int} 映射字典
        """
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            raise ImportError(
                "leidenalg / igraph 未安装，请运行: pip install leidenalg igraph"
            )

        if len(G.nodes) == 0:
            return {}

        # 1. 运行时密度监测 (Weighted Average Degree)
        total_weight = G.size(weight='weight')
        num_nodes = len(G.nodes)
        avg_degree = (2.0 * total_weight) / num_nodes if num_nodes > 0 else 0

        # 2. 动态 resolution 计算（与原 Louvain 版本相同公式，保持连续性）
        import math
        d_base = 3.0
        alpha = 0.5
        if avg_degree > 0:
            resolution = 1.0 + alpha * math.log(1 + max(0, avg_degree - d_base) / d_base)
        else:
            resolution = 1.0
        resolution = max(0.5, min(2.5, resolution))

        # 3. networkx → igraph 转换
        # 必须在转换后从 igraph 取顶点名，而非事先用 list(G.nodes())，
        # 因为 ig.Graph.from_networkx 内部顶点顺序不保证与 nx 迭代顺序一致
        ig_graph = ig.Graph.from_networkx(G)
        node_list = ig_graph.vs["_nx_name"]   # igraph 保存原始 NX 节点键
        weights = ig_graph.es['weight'] if ig_graph.ecount() > 0 else None

        # 4. Leiden 聚类（seed=42 保证可复现）
        try:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                seed=42,
                resolution_parameter=resolution,
            )
        except (ValueError, RuntimeError) as e:
            print(f"⚠ Leiden 聚类失败（{e}），跳过社区重建")
            return {}

        # 5. 映射回 {node_id_str: community_int}
        return {node_list[i]: comm for i, comm in enumerate(partition.membership)}

    def _generate_community_summary(
        self, community_id: str, member_ids: List[str]
    ) -> CommunitySummary:
        """
        调用 LLM 为一个社区生成主题标题、摘要和关键词。

        参数:
            community_id: 社区 ID 字符串
            member_ids: 该社区内所有 note 的 id 列表

        返回:
            CommunitySummary 对象
        """
        members = [self.memories[mid] for mid in member_ids if mid in self.memories]
        if not members:
            return CommunitySummary(
                community_id=community_id, level=0, member_note_ids=member_ids
            )

        # 构造 Prompt 内容
        # 优化：引入时序信息，确保事件逻辑顺序正确
        members = sorted(members, key=lambda m: m.timestamp)
        notes_text = ""
        for m in members[:50]:   # 放宽到前50条，充分利用现代大模型上下文窗口
            notes_text += (
                f"[{m.timestamp}] {m.content} "
                f"(keywords: {', '.join(m.keywords[:5])})\n"
            )

        prompt = f"""You are a memory clustering expert. Below are {min(len(members), 50)} memory fragments (out of {len(members)} total in this cluster) belonging to the same topic cluster, ordered by time:

{notes_text}

Analyze this group of memories and generate a comprehensive JSON summary. Your analysis must capture the core topic, main events, key entities, and chronological timeline.

Please ensure:
1. "title": A single sentence summarizing the core topic (max 10 words).
2. "summary": A comprehensive summary of the memories (max 100 words), covering the main events, entities, and the timeline.
3. "keywords": Provide exactly 5 to 10 representative and unique keywords that best describe this cluster.

Respond strictly in valid JSON format matching the required schema."""

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "community_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title":    {"type": "string"},
                        "summary":  {"type": "string"},
                        "keywords": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Provide exactly 5 to 10 representative keywords."
                        },
                    },
                    "required": ["title", "summary", "keywords"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        title, summary, keywords = "", "", []
        try:
            raw = self.llm_controller.llm.get_completion(prompt, response_format=response_format)
            raw_clean = raw.strip()
            if not raw_clean.startswith("{"):
                raw_clean = raw_clean[raw_clean.find("{"):]
            if not raw_clean.endswith("}"):
                raw_clean = raw_clean[: raw_clean.rfind("}") + 1]
            parsed = json.loads(raw_clean)
            title = parsed.get("title", "")
            summary = parsed.get("summary", "")
            keywords = parsed.get("keywords", [])
        except Exception as e:
            print(f"⚠ 社区摘要生成失败 ({community_id}): {e}")
            if 'raw' in locals():
                print(f"LLM 原始输出: {raw}")

        # 对摘要文本生成 embedding
        embedding = None
        if summary:
            try:
                emb = self.retriever.model.encode([summary])
                if hasattr(emb, 'numpy'):
                    emb = emb.numpy()
                # 确保保存的社区 embedding 是一维的 numpy 数组
                if len(emb.shape) == 2:
                    emb = emb[0]
                embedding = emb
            except Exception as e:
                print(f"⚠ 社区 embedding 生成失败: {e}")

        return CommunitySummary(
            community_id=community_id,
            level=0,
            member_note_ids=member_ids,
            title=title,
            summary=summary,
            keywords=keywords,
            embedding=embedding,
            updated_at=datetime.now().strftime("%Y%m%d%H%M"),
        )

    def _match_old_communities(
        self,
        old_communities: Dict[str, "CommunitySummary"],
        new_member_groups: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        用 Jaccard 相似度将新社区映射到最相近的旧社区。

        参数:
            old_communities: 重建前的 self.communities 快照
            new_member_groups: {new_community_id -> member_note_ids}

        返回:
            {new_community_id -> old_community_id}（无匹配则不含该键）

        注意：无唯一性约束，同一旧社区可被多个新社区匹配（社区分裂场景）。
        此时被分裂的 notes 会同时出现在多个新社区的 added_note_ids 中，
        也会出现在对方的 removed_note_ids 里（相对旧社区集合的 diff 是正确的）。
        """
        matches: Dict[str, str] = {}
        for new_cid, new_members in new_member_groups.items():
            new_set = set(new_members)
            if not new_set:
                continue
            best_old_cid, best_score = None, 0.0
            for old_cid, old_cs in old_communities.items():
                old_set = set(old_cs.member_note_ids)
                if not old_set:
                    continue
                intersection = len(new_set & old_set)
                union = len(new_set | old_set)
                jaccard = intersection / union  # both sets non-empty, union >= 1
                if jaccard > best_score:
                    best_score = jaccard
                    best_old_cid = old_cid
            if best_old_cid is not None and best_score >= 0.3:
                matches[new_cid] = best_old_cid
        return matches

    def rebuild_communities(self) -> None:
        """
        GraphRAG 社区重建：
          1. 构建 networkx 图
          2. 运行 Leiden 社区检测
          3. 为每个社区生成 LLM 摘要
          4. 更新 self.communities 和 community_embeddings
          5. 记录社区演化历史（evolution_history）

        会自动更新每条 note 的 community_id 字段。
        """
        if len(self.memories) < 2:
            return

        if not self._graph_is_dirty and len(self.communities) > 0:
            print(f"⏩ GraphRAG: 图结构未变脏，跳过社区重建")
            return

        print(f"🔄 GraphRAG: 重建社区（共 {len(self.memories)} 条记忆）...")

        # 快照旧社区，用于演化 diff
        old_communities: Dict[str, CommunitySummary] = dict(self.communities)

        # Step 1: 建图
        try:
            G = self._build_networkx_graph()
        except ImportError as e:
            print(f"⚠ {e}，跳过社区重建")
            return

        # Step 2: 社区检测
        try:
            partition = self._run_community_detection(G)   # {note_id: int}
        except ImportError as e:
            print(f"⚠ {e}，跳过社区重建")
            return

        if not partition:
            return

        # Step 3: 按社区 id 分组（排除 entity 虚拟节点）
        community_groups: Dict[int, List[str]] = {}
        for node_id, comm_int in partition.items():
            if not node_id.startswith("ent:"):
                community_groups.setdefault(comm_int, []).append(node_id)

        # 构建 new_cid -> member_ids 映射（用于匹配和 diff 计算）
        new_member_groups: Dict[str, List[str]] = {
            f"comm_{comm_int:04d}": member_ids
            for comm_int, member_ids in community_groups.items()
        }

        # Step 4: 将新社区匹配到旧社区（用于演化追踪）
        cid_to_old = self._match_old_communities(old_communities, new_member_groups)

        rebuild_ts = datetime.now().strftime("%Y%m%d%H%M")
        total_notes = len(self.memories)

        # Step 5: 为每个社区生成摘要并记录演化历史
        new_communities: Dict[str, CommunitySummary] = {}
        all_embeddings = []
        cid_order = []

        for comm_int, member_ids in community_groups.items():
            community_id = f"comm_{comm_int:04d}"

            # 更新 note.community_id
            for mid in member_ids:
                if mid in self.memories:
                    self.memories[mid].community_id = community_id

            # 生成摘要
            cs = self._generate_community_summary(community_id, member_ids)

            # 计算演化 diff
            old_cid = cid_to_old.get(community_id)
            if old_cid and old_cid in old_communities:
                old_cs = old_communities[old_cid]
                old_set = set(old_cs.member_note_ids)
                new_set = set(member_ids)
                evolution_entry = {
                    "timestamp": rebuild_ts,
                    "total_notes": total_notes,
                    "added_note_ids": sorted(new_set - old_set),
                    "removed_note_ids": sorted(old_set - new_set),
                    "prev_title": old_cs.title,
                }
                cs.evolution_history = old_cs.evolution_history + [evolution_entry]
            else:
                # 新出现的社区，记录首次创建
                cs.evolution_history = [{
                    "timestamp": rebuild_ts,
                    "total_notes": total_notes,
                    "added_note_ids": sorted(member_ids),
                    "removed_note_ids": [],
                    "prev_title": "",
                }]

            new_communities[community_id] = cs

            if cs.embedding is not None:
                all_embeddings.append(cs.embedding)
                cid_order.append(community_id)

        # Step 6: 更新类属性
        self.communities = new_communities
        self.community_ids_list = cid_order
        if all_embeddings:
            self.community_embeddings = np.stack(all_embeddings, axis=0)
        else:
            self.community_embeddings = None

        # 重建完成后，重置脏标记
        self._graph_is_dirty = False

        print(f"✅ GraphRAG: 社区重建完成，共 {len(new_communities)} 个社区：")
        for cid, cs in new_communities.items():
            n_added = len(cs.evolution_history[-1]["added_note_ids"]) if cs.evolution_history else 0
            n_removed = len(cs.evolution_history[-1]["removed_note_ids"]) if cs.evolution_history else 0
            print(f"   [{cid}] {cs.title} ({len(cs.member_note_ids)} 条记忆, +{n_added}/-{n_removed})")

    def find_community_context(self, query: str, k: int = 3) -> List[CommunitySummary]:
        """
        Global Search：对查询，检索最相关的 Top-K 社区摘要。
        适合回答宏观问题，如"我们都讨论过哪些话题"。

        参数:
            query: 查询文本
            k: 返回的社区数量

        返回:
            CommunitySummary 列表（按相关性降序）
        """
        if not self.communities or self.community_embeddings is None:
            return []

        query_emb = self.retriever.model.encode([query])   # shape (1, D)
        sims = cosine_similarity(query_emb, self.community_embeddings)[0]

        k = min(k, len(self.community_ids_list))
        top_indices = np.argsort(sims)[-k:][::-1]

        return [
            self.communities[self.community_ids_list[i]]
            for i in top_indices
            if self.community_ids_list[i] in self.communities
        ]

    def find_related_memories_with_community(
        self, query: str, k: int = 5
    ) -> Dict[str, Any]:
        """
        Local Search 增强版：检索最相关的 notes，并附带其所属社区的摘要作为背景。
        原有 find_related_memories() 接口不变，此为新增增强入口。

        参数:
            query: 查询文本
            k: 返回的 note 数量

        返回:
            字典，包含：
              - notes: List[MemoryNote]  相关记忆列表
              - community_context: str   社区摘要背景（格式化字符串）
              - communities: List[CommunitySummary]  涉及的社区列表
        """
        # Step 1: 原生 Local Search
        indices = self._retrieve_indices(query, k)
        related_notes = [self._get_note_by_doc_idx(i) for i in indices if self._get_note_by_doc_idx(i)]

        # Step 2: 收集这些 notes 涉及的社区
        involved_community_ids = set()
        for note in related_notes:
            if note.community_id and note.community_id in self.communities:
                involved_community_ids.add(note.community_id)

        involved_communities = [
            self.communities[cid] for cid in involved_community_ids
        ]

        # Step 3: 格式化社区背景
        community_context = ""
        for cs in involved_communities:
            kws_str = ", ".join(cs.keywords[:5]) if cs.keywords else "None"
            community_context += (
                f"[{cs.title}] {cs.summary} "
                f"(Keywords: {kws_str})\n"
            )

        return {
            "notes": related_notes,
            "community_context": community_context.strip(),
            "communities": involved_communities,
        }

    # ══════════════════════════════════════════════════════════════════════
    # ██  检索升级增强模块：HyDE + 社区感知 + PPR 多跳
    # ══════════════════════════════════════════════════════════════════════

    def _generate_hyde_query(self, query: str) -> str:
        """
        [HyDE] 生成假设性回复
        用原始提问生成一个拟真的假答案，将问句转为陈述句，以对抗 Dense 检索时的 Query-Doc 不对称性。
        """
        prompt = f"""You are an AI generating hypothetical documents.
The user is asking a question about a technical discussion or memory log. 
Please write 1 to 2 sentences of a plausible answer to this question. Do not state that you don't know. Just guess a highly realistic, plausible answer that contains relevant terminology.

User Query: {query}

Output JUST the hypothetical answer text, nothing else."""
        try:
            hyde_doc = self.llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "text"},
                temperature=0.7
            )
            # Combine the query and the hypothetical document for a richer embedding
            return f"{query} {hyde_doc.strip()}"
        except Exception as e:
            print(f"⚠ HyDE generation failed: {e}. Falling back to original query.")
            return query

    def _community_filter(self, query_emb: np.ndarray, top_c: int = 2) -> set:
        """
        [Community Filter] 层级检索：先定位 Top 社区
        返回命中社区内的 member_note_ids 集合。
        """
        if not self.communities or self.community_embeddings is None:
            return None
            
        # 防御性转换：确保是 NumPy 数组且为二维 (1, D)
        if hasattr(query_emb, 'numpy'):
            query_emb = query_emb.numpy()
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
            
        sims = cosine_similarity(query_emb, self.community_embeddings)[0]
        k = min(top_c, len(self.community_ids_list))
        top_indices = np.argsort(sims)[-k:][::-1]
        
        target_ids = set()
        for i in top_indices:
            cid = self.community_ids_list[i]
            if cid in self.communities:
                for mid in self.communities[cid].member_note_ids:
                    target_ids.add(mid)
                    
        return target_ids

    def _ppr_walk(self, seed_ids: list, query: str = "", damping: float = 0.8, max_hops: int = 2) -> dict:
        """
        [PPR Multi-hop] 个性化页面排名（多跳漫游）
        从种子节点出发，利用 semantic, temporal, Jaccard 及 LLM 提取的因果时序等图边扩展隐藏线索。
        引入 Query 意图感知：根据 query 内容动态调整边权重。
        """
        from collections import defaultdict
        scores = defaultdict(float)
        
        # 简单 Query 意图探测（词汇级触发）
        q_lower = query.lower()
        find_cause = any(w in q_lower for w in ["why", "cause", "reason", "because", "how come", "导致", "原因", "为什么", "起因"])
        find_effect = any(w in q_lower for w in ["result", "effect", "outcome", "happen next", "then", "结果", "影响", "导致了", "接下来"])
        
        # 种子节点初始分数
        for sid in seed_ids:
            scores[sid] = 1.0
            
        # 预加载邻接表，支持多关系和有向权重差异化
        adj = defaultdict(lambda: defaultdict(float))
        
        # 基础关系权重乘子
        type_multipliers = {
            "causes": 1.5,
            "results_in": 1.5,
            "happens_before": 1.2,
            "happens_after": 1.2,
            "temporal_entity_evolution": 1.2,
            "semantic": 1.0,
            "temporal_adjacent": 0.8,
            "entity_shared": 0.8,
            "lexical_shared": 1.2, # BM25 强匹配的专有名词
            "similar_to": 1.0,
            "llm_inferred": 1.2
        }

        for edge in self.graph_edges:
            src, tgt, w = edge.get("src_id"), edge.get("tgt_id"), edge.get("weight", 0.1)
            rel_type = edge.get("type", "semantic")
            
            multiplier = type_multipliers.get(rel_type, 1.0)
            
            # 【方案4】基于时间差的衰减因子
            # 只有当两个端点都是 MemoryNote 且有时间戳时才计算
            time_decay = 1.0
            if src in self.memories and tgt in self.memories:
                src_note, tgt_note = self.memories[src], self.memories[tgt]
                if src_note.timestamp and tgt_note.timestamp and len(src_note.timestamp) >= 8 and len(tgt_note.timestamp) >= 8:
                    try:
                        import math
                        from datetime import datetime
                        # 简化计算：只比较天数差异。YYYYMMDD
                        src_day = datetime.strptime(src_note.timestamp[:8], "%Y%m%d")
                        tgt_day = datetime.strptime(tgt_note.timestamp[:8], "%Y%m%d")
                        diff_days = abs((src_day - tgt_day).days)
                        # 指数衰减：半衰期约 7 天 (e^-0.1*7 ≈ 0.5)
                        time_decay = max(0.2, math.exp(-0.1 * diff_days))
                    except ValueError:
                        pass
            
            adjusted_w = w * multiplier * time_decay
            
            # Query-Aware 动态有向权重衰减
            if rel_type in ["causes", "results_in"]:
                # 如果查询在找原因，那么“溯源”方向（即从结果往起因跳）不应该被过度衰减
                forward_decay = 1.0
                backward_decay = 0.5
                if find_cause and rel_type == "causes": # src causes tgt. src is cause. tgt to src should be strong.
                    forward_decay, backward_decay = 0.5, 1.2
                elif find_effect and rel_type == "causes":
                    forward_decay, backward_decay = 1.2, 0.3
                    
                adj[src][tgt] = max(adj[src][tgt], adjusted_w * forward_decay)
                adj[tgt][src] = max(adj[tgt][src], adjusted_w * backward_decay)
                
            elif rel_type in ["happens_before", "temporal_entity_evolution"]:
                forward_decay, backward_decay = 1.0, 0.5
                if find_cause: # 找原因往往要往过去找
                    forward_decay, backward_decay = 0.6, 1.2
                adj[src][tgt] = max(adj[src][tgt], adjusted_w * forward_decay)
                adj[tgt][src] = max(adj[tgt][src], adjusted_w * backward_decay)
            else:
                adj[src][tgt] = max(adj[src][tgt], adjusted_w)
                adj[tgt][src] = max(adj[tgt][src], adjusted_w)
            
        current_frontier = set(seed_ids)
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            for node in current_frontier:
                current_score = scores[node]
                for neighbor, weight in adj[node].items():
                    # 传播得分: 节点现有得分 * 边权重 * 衰减率
                    # 限制单次传播的最大比例，防止得分无限放大导致远距离节点反超种子节点
                    effective_weight = min(float(weight), 1.2)
                    propagated = current_score * effective_weight * damping
                    
                    # 取历史得分和传播得分的较大值，但为了避免得分无限制膨胀，增加对种子节点的降级约束
                    if propagated > scores[neighbor]:
                        # 强迫非种子节点（或者传播过程中）的得分严格小于或等于源节点，确保符合漫游衰减规律
                        capped_propagated = min(propagated, current_score * 0.95)
                        if capped_propagated > scores[neighbor]:
                            scores[neighbor] = capped_propagated
                            next_frontier.add(neighbor)
            current_frontier = next_frontier
            
        return dict(scores)

    def find_related_memories_advanced(
        self,
        query: str,
        k: int = 5,
        include_community_context: bool = True,
        max_hops: int = 2,
        alpha: float = 0.5,
        use_hyde: bool = False,
        dense_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        终极检索演进管道：Community Filter + BGE-M3 + PPR Walk
        返回含有丰富线索组合的综合证据字典。
        完全兼容原有的返回结构。

        use_hyde: 是否启用 HyDE 假设文档扩展。
                  对话记忆封闭世界检索默认关闭（LLM 无法预知私有事件的具体细节，
                  生成的假设文档会引入幻觉偏移）；仅在开放域复杂推理场景下考虑开启。
        """
        all_memories_list = list(self.memories.values())
        if not all_memories_list:
             return {"notes": [], "community_context": "", "communities": []}

        # 1. HyDE（默认关闭：对话记忆为封闭世界，假设文档会引入幻觉偏移）
        if use_hyde:
            hyde_doc = self._generate_hyde_query(query)
        else:
            hyde_doc = query

        # 2. Vectorize
        query_emb = self.retriever.model.encode([hyde_doc])
        
        # 3. Community Filter
        # 稍微放宽 top_c 以减少漏网之鱼，并将硬过滤完全变为软过滤
        allowed_ids = self._community_filter(query_emb, top_c=3)
        
        # 4. Hybrid（Dense + BM25）精搜种子节点
        scores = self.retriever.get_hybrid_scores(
            query=query,
            query_embedding=query_emb[0],
            dense_weight=dense_weight,
        )
        
        if len(scores) == 0:
            return {"notes": [], "community_context": "", "communities": []}
            
        seed_scores = []
        for i, doc_str in enumerate(self.retriever.corpus):
            mem = self._get_note_by_doc_idx(i)
            if mem:
                score = float(scores[i])
                # 将社区过滤改为软过滤（加分），避免硬过滤导致高相关性节点丢失
                if allowed_ids is not None and mem.id in allowed_ids:
                    score += 0.15 # 提高社区加分权重，引导聚类但不仅限于聚类
                seed_scores.append((mem.id, score))
                    
        # 提取 Top-K 种子
        seed_scores.sort(key=lambda x: x[1], reverse=True)
        # 控制种子规模，避免 PPR 扩散过度带来噪声
        seed_cap = min(max(k, 16), 32)
        top_seeds = [t[0] for t in seed_scores[:seed_cap]]
        
        # 5. PPR 多跳（可按题型降 hop，减少噪音扩散）
        if top_seeds:
            ppr_results = self._ppr_walk(
                top_seeds,
                query=query,
                damping=0.8,
                max_hops=max(0, int(max_hops)),
            )
        else:
            ppr_results = {}
            
        # 6. 合并收网 (动态归一化融合 Dense Score 和 PPR Score)
        # 建立 Dense 得分字典，仅考虑种子节点和PPR走到的节点，减少计算量
        candidate_ids = set(top_seeds) | set(ppr_results.keys())
        dense_scores_dict = {mem_id: sc for mem_id, sc in seed_scores if mem_id in candidate_ids}
        
        # 获取分数的 Min-Max 以便归一化
        def min_max_norm(score_dict):
            if not score_dict:
                return {}
            vals = list(score_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v - min_v < 1e-6:
                return {k: 1.0 for k in score_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}
            
        norm_dense = min_max_norm(dense_scores_dict)
        norm_ppr = min_max_norm(ppr_results)
        
        # 融合打分 S_final = alpha * Dense + (1-alpha) * PPR
        alpha = max(0.0, min(1.0, float(alpha)))
        fused_scores = {}
        
        for mid in candidate_ids:
            if mid not in self.memories:
                continue
            # 修正未命中时的默认得分。因为min_max_norm的最小值为0，所以未命中的节点应得0.0而不是0.05
            d_score = norm_dense.get(mid, 0.0)
            p_score = norm_ppr.get(mid, 0.0)
            
            fused_scores[mid] = alpha * d_score + (1 - alpha) * p_score
            
        final_list = [(self.memories[mid], sc) for mid, sc in fused_scores.items() if sc > 0.01]
        final_list.sort(key=lambda x: x[1], reverse=True)
        
        # 限制最终注入给生成模型的证据条数，避免长上下文导致答案漂移
        note_cap = min(k, 20)
        # 取 2x 候选送入 reranker 精排，再截断到 note_cap
        candidate_notes = [x[0] for x in final_list[:note_cap * 2]]
        reranker = self._get_reranker() if self.use_reranking else None
        if reranker is not None and len(candidate_notes) > note_cap:
            pairs = [[query, f"{n.content} {n.context}"] for n in candidate_notes]
            try:
                rr_scores = reranker.predict(pairs)
                ranked = [candidate_notes[i] for i in np.argsort(rr_scores)[::-1]]
                related_notes = ranked[:note_cap]
            except Exception:
                related_notes = candidate_notes[:note_cap]
        else:
            related_notes = candidate_notes[:note_cap]
        
        # 如果依然不够，回退填充
        if len(related_notes) < note_cap and seed_scores:
            existing_ids = set(n.id for n in related_notes)
            for t in seed_scores:
                if t[0] not in existing_ids and t[0] in self.memories:
                    related_notes.append(self.memories[t[0]])
                if len(related_notes) >= note_cap:
                    break
                     
        # 7. 拼接社区语境 (Community Context)
        involved_community_ids = set()
        for note in related_notes:
            if note.community_id and note.community_id in self.communities:
                involved_community_ids.add(note.community_id)

        involved_communities = [self.communities[cid] for cid in involved_community_ids] if include_community_context else []

        community_context = ""
        if include_community_context:
            for cs in involved_communities:
                kws_str = ", ".join(cs.keywords[:5]) if cs.keywords else "None"
                community_context += (
                    f"[{cs.title}] {cs.summary} "
                    f"(Keywords: {kws_str})\n"
                )
                # 注入最近演化历史，帮助 LLM 识别已更新/过期的信息
                # 对 temporal-reasoning 和 knowledge-update 类问题有直接收益
                if cs.evolution_history:
                    for ev in cs.evolution_history[-2:]:
                        removed_ids = ev.get("removed_note_ids", [])
                        if removed_ids:
                            snippets = []
                            for nid in removed_ids[:3]:
                                n = self.memories.get(nid)
                                if n:
                                    snippets.append(n.content[:80])
                            if snippets:
                                ts_short = ev.get("timestamp", "")[:8]
                                community_context += (
                                    f"  [知识更新@{ts_short}] 以下信息已被新内容替代: "
                                    + " | ".join(snippets) + "\n"
                                )

        return {
            "notes": related_notes,
            "community_context": community_context.strip(),
            "communities": involved_communities,
        }

    # ══════════════════════════════════════════════════════════════════════
    # ██  Multi-Channel Parallel Retrieval with RRF Fusion
    # ══════════════════════════════════════════════════════════════════════

    def retrieve_multi_channel_rrf(
        self,
        query: str,
        k: int = 10,
        program: Optional[EvidenceProgram] = None,
        channel_weights: Optional[Dict[str, float]] = None,
        k_rrf: int = 60,
        enable_ppr: bool = False,
        ppr_max_hops: int = 2,
        ppr_damping: float = 0.8,
        include_community_context: bool = True,
        community_top_c: int = 5,
    ) -> Dict[str, Any]:
        """
        多通道并行检索 + RRF 融合。

        四个独立检索通道：
          - Dense:     BGE-M3 cosine 语义检索
          - BM25:      词汇级稀疏检索
          - Community: 社区 embedding 匹配 → 成员展开 → 二次排序（创新通道）
          - PPR:       图上个性化 PageRank 多跳漫游（可选）

        通过 Reciprocal Rank Fusion 合并各通道排名，
        channel_weights 控制各通道对最终得分的贡献。

        返回值与 find_related_memories_advanced 兼容。
        """
        from collections import defaultdict

        all_memories_list = list(self.memories.values())
        if not all_memories_list:
            return {"notes": [], "community_context": "", "communities": [],
                    "rrf_scores": {}, "channel_details": {}}

        default_weights = {"dense": 1.0, "bm25": 0.8, "community": 0.5, "ppr": 0.0}
        weights = channel_weights or default_weights

        # ── Step A: 编码 query（只做一次） ──────────────────────────
        query_emb = self.retriever.model.encode([query])
        if hasattr(query_emb, 'numpy'):
            query_emb = query_emb.numpy()
        query_emb_1d = query_emb[0] if len(query_emb.shape) == 2 else query_emb
        query_emb_2d = query_emb_1d.reshape(1, -1)

        dense_top_n = 50
        bm25_top_n = 50
        community_top_n = 30

        # ── Step B: Channel 1 — Dense Retrieval ───────────────────
        dense_ranked: List[Tuple[str, int]] = []
        if weights.get("dense", 0) > 0:
            dense_scores = self.retriever.get_scores_by_emb(query_emb_1d)
            if len(dense_scores) > 0:
                top_n = min(dense_top_n, len(dense_scores))
                top_indices = np.argsort(dense_scores)[-top_n:][::-1]
                rank = 1
                for idx in top_indices:
                    mem = self._get_note_by_doc_idx(idx)
                    if mem:
                        dense_ranked.append((mem.id, rank))
                        rank += 1

        # ── Step C: Channel 2 — BM25 Retrieval ────────────────────
        bm25_ranked: List[Tuple[str, int]] = []
        if weights.get("bm25", 0) > 0 and self.retriever.bm25 is not None:
            query_tokens = self.retriever._tokenize_for_bm25(query)
            bm25_scores = np.asarray(
                self.retriever.bm25.get_scores(query_tokens), dtype=float
            )
            if len(bm25_scores) > 0:
                top_n = min(bm25_top_n, len(bm25_scores))
                top_indices = np.argsort(bm25_scores)[-top_n:][::-1]
                rank = 1
                for idx in top_indices:
                    if bm25_scores[idx] <= 0:
                        continue
                    mem = self._get_note_by_doc_idx(idx)
                    if mem:
                        bm25_ranked.append((mem.id, rank))
                        rank += 1

        # ── Step D: Channel 3 — Community Retrieval（创新通道）─────
        community_ranked: List[Tuple[str, int]] = []
        if (weights.get("community", 0) > 0
                and self.communities
                and self.community_embeddings is not None
                and community_top_c > 0):
            comm_sims = cosine_similarity(query_emb_2d, self.community_embeddings)[0]
            top_c = min(community_top_c, len(self.community_ids_list))
            top_comm_indices = np.argsort(comm_sims)[-top_c:][::-1]

            # 展开成员 note IDs（去重）
            candidate_note_ids: List[str] = []
            seen_ids: set = set()
            for ci in top_comm_indices:
                cid = self.community_ids_list[ci]
                if cid in self.communities:
                    for mid in self.communities[cid].member_note_ids:
                        if mid not in seen_ids:
                            seen_ids.add(mid)
                            candidate_note_ids.append(mid)

            # 按 dense 相似度对展开成员排序
            member_scores: List[Tuple[str, float]] = []
            for mid in candidate_note_ids:
                if mid not in self.memories:
                    continue
                doc_text = self._generate_doc_text(self.memories[mid])
                doc_idx = self.retriever.document_ids.get(doc_text)
                if doc_idx is not None and doc_idx < len(self.retriever.embeddings):
                    sim = float(cosine_similarity(
                        query_emb_2d,
                        self.retriever.embeddings[doc_idx].reshape(1, -1)
                    )[0][0])
                    member_scores.append((mid, sim))

            member_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (mid, _) in enumerate(member_scores[:community_top_n], 1):
                community_ranked.append((mid, rank))

        # ── Step E: Channel 4 — PPR Graph Walk（可选）──────────────
        ppr_ranked: List[Tuple[str, int]] = []
        if enable_ppr and weights.get("ppr", 0) > 0 and self.graph_edges:
            seed_ids: set = set()
            for ranked_list in [dense_ranked[:10], bm25_ranked[:10], community_ranked[:10]]:
                for (nid, _) in ranked_list:
                    seed_ids.add(nid)
            if seed_ids:
                ppr_results = self._ppr_walk(
                    list(seed_ids), query=query,
                    damping=ppr_damping, max_hops=max(0, int(ppr_max_hops)),
                )
                sorted_ppr = sorted(ppr_results.items(), key=lambda x: x[1], reverse=True)
                for rank, (mid, _) in enumerate(sorted_ppr, 1):
                    if mid in self.memories:
                        ppr_ranked.append((mid, rank))

        # ── Step F: RRF Fusion ─────────────────────────────────────
        rrf_scores: Dict[str, float] = defaultdict(float)
        channel_data = {
            "dense": dense_ranked,
            "bm25": bm25_ranked,
            "community": community_ranked,
            "ppr": ppr_ranked,
        }
        for channel_name, ranked_list in channel_data.items():
            w = weights.get(channel_name, 0.0)
            if w <= 0 or not ranked_list:
                continue
            for (note_id, rank) in ranked_list:
                rrf_scores[note_id] += w / (k_rrf + rank)

        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        constraints = dict((program.metadata or {}).get("answer_constraints", {})) if program is not None else {}
        if constraints and sorted_fused:
            rerank_pool_size = min(len(sorted_fused), max(k * 4, 24))
            reranked_pool = self._apply_constraint_rerank(
                query,
                sorted_fused[:rerank_pool_size],
                constraints=constraints,
            )
            sorted_fused = reranked_pool + sorted_fused[rerank_pool_size:]

        # ── Step G: Optional Reranking ──────────────────────────────
        candidate_pool_size = min(len(sorted_fused), max(k * 3, 18))
        candidate_ids = [mid for mid, _ in sorted_fused[:candidate_pool_size]]
        candidate_notes = [self.memories[mid] for mid in candidate_ids if mid in self.memories]

        reranker = self._get_reranker() if self.use_reranking else None
        if reranker is not None and len(candidate_notes) > k:
            pairs = [[query, f"{n.content} {n.context}"] for n in candidate_notes]
            try:
                rr_scores = reranker.predict(pairs)
                ranked = [candidate_notes[i] for i in np.argsort(rr_scores)[::-1]]
                related_notes = ranked[:k]
            except Exception:
                related_notes = candidate_notes[:k]
        else:
            related_notes = candidate_notes[:k]

        # ── Step H: 社区上下文 ──────────────────────────────────────
        involved_community_ids: set = set()
        for note in related_notes:
            if note.community_id and note.community_id in self.communities:
                involved_community_ids.add(note.community_id)

        community_context = ""
        involved_communities: list = []
        if include_community_context:
            involved_communities = [self.communities[cid] for cid in involved_community_ids]
            for cs in involved_communities:
                kws_str = ", ".join(cs.keywords[:5]) if cs.keywords else "None"
                community_context += (
                    f"[{cs.title}] {cs.summary} "
                    f"(Keywords: {kws_str})\n"
                )
                if cs.evolution_history:
                    for ev in cs.evolution_history[-2:]:
                        removed_ids = ev.get("removed_note_ids", [])
                        if removed_ids:
                            snippets = []
                            for nid in removed_ids[:3]:
                                n = self.memories.get(nid)
                                if n:
                                    snippets.append(n.content[:80])
                            if snippets:
                                ts_short = ev.get("timestamp", "")[:8]
                                community_context += (
                                    f"  [Knowledge Update @{ts_short}] Superseded: "
                                    + " | ".join(snippets) + "\n"
                                )

        return {
            "notes": related_notes,
            "community_context": community_context.strip(),
            "communities": involved_communities,
            "rrf_scores": dict(rrf_scores),
            "channel_details": {
                "dense_ranked": dense_ranked,
                "bm25_ranked": bm25_ranked,
                "community_ranked": community_ranked,
                "ppr_ranked": ppr_ranked,
            },
        }

    # ══════════════════════════════════════════════════════════════════════
    # ██  Multi-Channel RRF 结束
    # ══════════════════════════════════════════════════════════════════════

    def build_community_context_for_notes(
        self,
        notes: List[MemoryNote],
        top_c: int = 5,
    ) -> Dict[str, Any]:
        involved_community_ids: List[str] = []
        seen_ids = set()
        for note in notes:
            cid = getattr(note, "community_id", None)
            if cid and cid in self.communities and cid not in seen_ids:
                seen_ids.add(cid)
                involved_community_ids.append(cid)

        involved_community_ids = involved_community_ids[: max(0, int(top_c))]
        involved_communities = [self.communities[cid] for cid in involved_community_ids if cid in self.communities]

        community_context = ""
        for cs in involved_communities:
            kws_str = ", ".join(cs.keywords[:5]) if cs.keywords else "None"
            community_context += (
                f"[{cs.title}] {cs.summary} "
                f"(Keywords: {kws_str})\n"
            )
            if cs.evolution_history:
                for ev in cs.evolution_history[-2:]:
                    removed_ids = ev.get("removed_note_ids", [])
                    if not removed_ids:
                        continue
                    snippets = []
                    for nid in removed_ids[:3]:
                        note = self.memories.get(nid)
                        if note:
                            snippets.append(note.content[:80])
                    if snippets:
                        ts_short = ev.get("timestamp", "")[:8]
                        community_context += (
                            f"  [Knowledge Update @{ts_short}] Superseded: "
                            + " | ".join(snippets) + "\n"
                        )

        return {
            "community_context": community_context.strip(),
            "communities": involved_communities,
        }

    def _program_query_terms(self, query: str) -> List[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at",
            "by", "is", "are", "was", "were", "be", "been", "being", "did", "do", "does",
            "what", "which", "who", "when", "where", "why", "how", "this", "that", "it",
            "question", "conversation", "mentioned", "mention",
        }
        return [tok for tok in re.findall(r"[A-Za-z0-9_]+", (query or "").lower()) if tok not in stopwords]

    def _score_note_for_program(self, query: str, note: MemoryNote, program: EvidenceProgram) -> float:
        query_terms = set(self._program_query_terms(query))
        note_text = " ".join(
            [
                note.content or "",
                note.context or "",
                " ".join(note.keywords or []),
                " ".join(note.tags or []),
            ]
        ).lower()
        note_terms = set(self._program_query_terms(note_text))
        overlap = len(query_terms & note_terms)
        constraints = dict((program.metadata or {}).get("answer_constraints", {}))

        score = float(overlap)
        if constraints:
            named_entities = [str(name).lower() for name in constraints.get("named_entities", []) if str(name).strip()]
            role_terms = [str(role).lower() for role in constraints.get("role_terms", []) if str(role).strip()]
            entity_hits = sum(1 for name in named_entities if name in note_text)
            role_hits = sum(1 for role in role_terms if role in note_text)
            if entity_hits:
                score += min(0.8, 0.35 * entity_hits)
            elif constraints.get("entity_sensitive"):
                score -= 0.2
            if role_hits:
                score += min(0.4, 0.2 * role_hits)
            elif role_terms and constraints.get("entity_sensitive"):
                score -= 0.1
        if getattr(note, "timestamp", None) and program.program_type == ProgramType.TEMPORAL:
            score += 0.8
        if getattr(note, "community_id", None) and program.include_community_context:
            score += 0.2
        if program.program_type == ProgramType.MULTI_HOP and getattr(note, "id_based_links", None):
            score += min(0.6, 0.15 * len(note.id_based_links))
        if program.program_type == ProgramType.PROFILE and (note.context or "").lower() not in {"", "general"}:
            score += 0.4
        return score

    def _dedupe_notes_by_id(self, notes: List[MemoryNote]) -> List[MemoryNote]:
        deduped: List[MemoryNote] = []
        seen_ids = set()
        seen_content = set()
        for note in notes:
            if note is None:
                continue
            note_id = getattr(note, "id", None)
            content = getattr(note, "content", None)
            if note_id and note_id in seen_ids:
                continue
            if not note_id and content and content in seen_content:
                continue
            deduped.append(note)
            if note_id:
                seen_ids.add(note_id)
            if content:
                seen_content.add(content)
        return deduped

    def _note_text_for_program(self, note: MemoryNote) -> str:
        return " ".join(
            [
                getattr(note, "content", "") or "",
                getattr(note, "context", "") or "",
                " ".join(getattr(note, "keywords", []) or []),
                " ".join(getattr(note, "tags", []) or []),
            ]
        ).strip()

    def _constraint_match_score(
        self,
        query: str,
        note: MemoryNote,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> float:
        if note is None or not constraints:
            return 0.0

        note_text = self._note_text_for_program(note).lower()
        if not note_text:
            return 0.0

        named_entities = [str(name).lower() for name in constraints.get("named_entities", []) if str(name).strip()]
        role_terms = [str(role).lower() for role in constraints.get("role_terms", []) if str(role).strip()]
        query_terms = set(self._program_query_terms(query))
        note_terms = set(self._program_query_terms(note_text))

        overlap = len(query_terms & note_terms)
        entity_hits = sum(1 for name in named_entities if name in note_text)
        role_hits = sum(1 for role in role_terms if role in note_text)

        score = 0.0
        if overlap:
            score += min(0.9, 0.18 * overlap)
        if entity_hits:
            score += min(2.2, 0.85 * entity_hits)
        elif constraints.get("entity_sensitive") and named_entities:
            score -= 0.75
        if role_hits:
            score += min(1.0, 0.45 * role_hits)
        elif constraints.get("entity_sensitive") and role_terms:
            score -= 0.35

        answer_type = str(constraints.get("answer_type", "") or "")
        if answer_type == "location" and re.search(r"\b(from|in|at|near|to)\b", note_text):
            score += 0.35
        if answer_type == "count" and re.search(
            r"\b\d+\b|\bonce\b|\btwice\b|\bthree\b|\bfour\b|\bfive\b|\bsix\b|\bseven\b|\beight\b|\bnine\b|\bten\b",
            note_text,
        ):
            score += 0.35
        if constraints.get("expects_list") and (", " in note_text or " and " in note_text):
            score += 0.25
        if answer_type == "reason_phrase" and re.search(
            r"\b(because|since|due to|inspired by|motivated by|realized that|after)\b",
            note_text,
        ):
            score += 0.3
        return score

    def _apply_constraint_rerank(
        self,
        query: str,
        ranked_items: List[Tuple[str, float]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if not constraints or not ranked_items:
            return list(ranked_items)

        rescored: List[Tuple[str, float, float]] = []
        for note_id, base_score in ranked_items:
            note = self.memories.get(note_id)
            if note is None:
                continue
            adjusted_score = float(base_score) + 0.0045 * self._constraint_match_score(query, note, constraints)
            rescored.append((note_id, adjusted_score, float(base_score)))

        rescored.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [(note_id, base_score) for note_id, _, base_score in rescored]

    def _build_note_edge_index(
        self,
        notes: List[MemoryNote],
        allowed_types: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], Dict[str, Any]]]:
        note_ids = {
            getattr(note, "id", None)
            for note in notes
            if getattr(note, "id", None)
        }
        if not note_ids or not self.graph_edges:
            return {}, {}

        allowed = set(
            allowed_types
            or [
                "semantic",
                "entity_shared",
                "temporal_entity_evolution",
                "causes",
                "results_in",
                "lexical_shared",
                "temporal_adjacent",
                "llm_inferred",
            ]
        )
        adjacency_sets: Dict[str, set] = {}
        edge_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for edge in self.graph_edges:
            src_id = edge.get("src_id")
            tgt_id = edge.get("tgt_id")
            edge_type = str(edge.get("type", "semantic") or "semantic")
            if not src_id or not tgt_id or src_id not in note_ids or tgt_id not in note_ids:
                continue
            if src_id not in self.memories or tgt_id not in self.memories:
                continue
            if allowed and edge_type not in allowed:
                continue

            weight = float(edge.get("weight", 0.0) or 0.0)
            pair = tuple(sorted((src_id, tgt_id)))
            best_edge = edge_lookup.get(pair)
            if best_edge is None or weight > float(best_edge.get("weight", 0.0) or 0.0):
                edge_lookup[pair] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "weight": round(weight, 4),
                    "type": edge_type,
                }

            adjacency_sets.setdefault(src_id, set()).add(tgt_id)
            adjacency_sets.setdefault(tgt_id, set()).add(src_id)

        adjacency = {
            node_id: sorted(neighbors)
            for node_id, neighbors in adjacency_sets.items()
        }
        return adjacency, edge_lookup

    def _find_best_subgraph_path(
        self,
        seed_ids: List[str],
        adjacency: Dict[str, List[str]],
        note_scores: Dict[str, float],
        max_hops: int,
    ) -> List[str]:
        if len(seed_ids) < 2:
            return list(seed_ids)

        start_id = seed_ids[0]
        max_depth = max(1, int(max_hops or 0) + 1)
        for target_id in seed_ids[1:]:
            queue: List[Tuple[str, List[str], int]] = [(start_id, [start_id], 0)]
            seen_depth: Dict[str, int] = {start_id: 0}
            while queue:
                current_id, path, depth = queue.pop(0)
                if current_id == target_id:
                    return path
                if depth >= max_depth:
                    continue

                ranked_neighbors = sorted(
                    adjacency.get(current_id, []),
                    key=lambda node_id: (-note_scores.get(node_id, 0.0), node_id),
                )
                for neighbor_id in ranked_neighbors:
                    next_depth = depth + 1
                    if next_depth > max_depth:
                        continue
                    if neighbor_id in seen_depth and seen_depth[neighbor_id] <= next_depth:
                        continue
                    seen_depth[neighbor_id] = next_depth
                    queue.append((neighbor_id, path + [neighbor_id], next_depth))
        return []

    def build_answer_subgraph(
        self,
        query: str,
        notes: List[MemoryNote],
        program: EvidenceProgram,
    ) -> EvidenceSubgraph:
        deduped = self._dedupe_notes_by_id(notes)
        if not deduped:
            return EvidenceSubgraph(metadata={"reason": "no_notes"})

        query_terms = set(self._program_query_terms(query))
        ranked_candidates: List[Tuple[MemoryNote, float, int]] = []
        note_scores: Dict[str, float] = {}
        note_terms: Dict[str, set] = {}
        note_by_id: Dict[str, MemoryNote] = {}

        for note in deduped:
            note_id = getattr(note, "id", None)
            if not note_id:
                continue
            terms = set(self._program_query_terms(self._note_text_for_program(note)))
            overlap = len(query_terms & terms)
            score = self._score_note_for_program(query, note, program) + min(1.2, 0.25 * overlap)
            ranked_candidates.append((note, score, overlap))
            note_scores[note_id] = score
            note_terms[note_id] = terms
            note_by_id[note_id] = note

        if not ranked_candidates:
            return EvidenceSubgraph(metadata={"reason": "no_note_ids"})

        ranked_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
        seed_target = 2 if (
            program.program_type == ProgramType.MULTI_HOP
            or program.answer_style in {AnswerStyle.LIST_SPAN, AnswerStyle.SUMMARY_SHORT}
        ) else 1
        seed_ids: List[str] = []
        for note, _, overlap in ranked_candidates:
            note_id = getattr(note, "id", None)
            if not note_id or note_id in seed_ids:
                continue
            if overlap > 0 or len(seed_ids) < seed_target:
                seed_ids.append(note_id)
            if len(seed_ids) >= seed_target:
                break

        adjacency, edge_lookup = self._build_note_edge_index(deduped)
        selected_ids: List[str] = []
        if program.program_type == ProgramType.MULTI_HOP:
            if len(seed_ids) < 2:
                return EvidenceSubgraph(
                    seed_node_ids=seed_ids,
                    metadata={"reason": "insufficient_seeds", "candidate_note_count": len(deduped)},
                )
            selected_ids = self._find_best_subgraph_path(seed_ids, adjacency, note_scores, program.max_hops)
            if not selected_ids:
                return EvidenceSubgraph(
                    seed_node_ids=seed_ids,
                    metadata={"reason": "no_connecting_path", "candidate_note_count": len(deduped)},
                )
        else:
            if not seed_ids:
                return EvidenceSubgraph(metadata={"reason": "no_seed"})
            target_nodes = min(
                max(1, int(program.max_notes or 1)),
                3 if program.answer_style in {AnswerStyle.LIST_SPAN, AnswerStyle.SUMMARY_SHORT} else 2,
            )
            selected_ids = list(seed_ids[:target_nodes])
            for neighbor_id in sorted(
                adjacency.get(seed_ids[0], []),
                key=lambda node_id: (-note_scores.get(node_id, 0.0), node_id),
            ):
                if neighbor_id in selected_ids:
                    continue
                if note_scores.get(neighbor_id, 0.0) <= 0.0:
                    continue
                selected_ids.append(neighbor_id)
                if len(selected_ids) >= target_nodes:
                    break

        selected_ids = [node_id for node_id in selected_ids if node_id in note_by_id]
        if not selected_ids or len(selected_ids) > max(1, int(program.max_notes or 1)):
            return EvidenceSubgraph(
                seed_node_ids=seed_ids,
                metadata={"reason": "selection_out_of_budget", "candidate_note_count": len(deduped)},
            )

        selected_edges: List[Dict[str, Any]] = []
        selected_edge_keys = set()
        for idx in range(len(selected_ids) - 1):
            pair = tuple(sorted((selected_ids[idx], selected_ids[idx + 1])))
            edge = edge_lookup.get(pair)
            if not edge or pair in selected_edge_keys:
                continue
            selected_edge_keys.add(pair)
            selected_edges.append(dict(edge))

        covered_terms = set()
        for node_id in selected_ids:
            covered_terms.update(query_terms & note_terms.get(node_id, set()))
        query_coverage = (len(covered_terms) / max(len(query_terms), 1)) if query_terms else 0.0
        avg_node_score = (
            sum(note_scores.get(node_id, 0.0) for node_id in selected_ids) / max(len(selected_ids), 1)
        )
        avg_edge_weight = (
            sum(float(edge.get("weight", 0.0) or 0.0) for edge in selected_edges) / max(len(selected_edges), 1)
            if selected_edges else 0.0
        )
        bridge_node_ids = [node_id for node_id in selected_ids if node_id not in seed_ids]
        connected = len(selected_ids) <= 1 or len(selected_edges) >= max(1, len(selected_ids) - 1)
        confidence = 0.22
        confidence += 0.34 * min(1.0, query_coverage)
        confidence += 0.22 * min(1.0, avg_edge_weight)
        confidence += 0.14 * min(1.0, avg_node_score / 4.0)
        if bridge_node_ids:
            confidence += 0.08
        confidence = round(min(0.99, max(0.0, confidence)), 4)

        valid = False
        reason = "low_support"
        top_overlap = max((len(query_terms & note_terms.get(node_id, set())) for node_id in selected_ids), default=0)
        if program.program_type == ProgramType.MULTI_HOP:
            valid = connected and len(selected_ids) >= 2 and bool(selected_edges) and query_coverage >= 0.12
            reason = "multi_hop_path" if valid else "multi_hop_gate_failed"
        elif program.program_type == ProgramType.VERIFY_UNSUPPORTED:
            valid = query_coverage >= 0.34 or (top_overlap >= 2 and (bool(selected_edges) or len(selected_ids) == 1))
            reason = "direct_support_cluster" if valid else "strict_support_gate_failed"
        elif program.answer_style == AnswerStyle.LIST_SPAN:
            valid = query_coverage >= 0.24 or (top_overlap >= 2 and len(selected_ids) >= 2)
            reason = "list_support_cluster" if valid else "list_support_gate_failed"
        elif program.answer_style == AnswerStyle.STRICT_ENTITY_SPAN:
            valid = query_coverage >= 0.18 or top_overlap >= 2
            reason = "strict_entity_cluster" if valid else "strict_entity_gate_failed"
        elif program.answer_style in {AnswerStyle.SUMMARY_SHORT, AnswerStyle.REASON_PHRASE}:
            valid = query_coverage >= 0.22 or (top_overlap >= 2 and len(selected_ids) >= 1)
            reason = "summary_support_cluster" if valid else "summary_support_gate_failed"

        metadata = {
            "reason": reason,
            "query_coverage": round(query_coverage, 4),
            "avg_node_score": round(avg_node_score, 4),
            "avg_edge_weight": round(avg_edge_weight, 4),
            "connected": connected,
            "path_length": len(selected_ids),
            "candidate_note_count": len(deduped),
        }
        return EvidenceSubgraph(
            nodes=[note_by_id[node_id] for node_id in selected_ids],
            edges=selected_edges,
            seed_node_ids=seed_ids,
            bridge_node_ids=bridge_node_ids,
            confidence=confidence,
            valid=valid,
            metadata=metadata,
        )

    def expand_evidence_graph(
        self,
        seed_notes: List[MemoryNote],
        max_hops: int = 2,
        edge_types: Optional[List[str]] = None,
    ) -> List[MemoryNote]:
        if not seed_notes or max_hops <= 0 or not self.graph_edges:
            return self._dedupe_notes_by_id(seed_notes)

        preferred_types = edge_types or [
            "semantic",
            "entity_shared",
            "temporal_entity_evolution",
            "causes",
            "results_in",
            "lexical_shared",
        ]
        allowed_types = set(preferred_types)
        type_rank = {edge_type: idx for idx, edge_type in enumerate(preferred_types)}
        adjacency: Dict[str, List[Tuple[int, float, str]]] = {}

        for edge in self.graph_edges:
            src_id = edge.get("src_id")
            tgt_id = edge.get("tgt_id")
            edge_type = edge.get("type", "semantic")
            if edge_type not in allowed_types:
                continue
            weight = float(edge.get("weight", 0.0))
            src_note = self.memories.get(src_id)
            tgt_note = self.memories.get(tgt_id)
            if src_note and tgt_note:
                adjacency.setdefault(src_id, []).append((type_rank.get(edge_type, 99), -weight, tgt_id))
                adjacency.setdefault(tgt_id, []).append((type_rank.get(edge_type, 99), -weight, src_id))

        max_nodes = max(len(seed_notes), min(10, len(seed_notes) + max(0, max_hops) * 2))
        ordered_notes = self._dedupe_notes_by_id(seed_notes)
        visited = {note.id for note in ordered_notes if getattr(note, "id", None)}
        frontier = [(note.id, 0) for note in ordered_notes if getattr(note, "id", None)]

        while frontier and len(ordered_notes) < max_nodes:
            current_id, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            for _, _, neighbor_id in sorted(adjacency.get(current_id, [])):
                if neighbor_id in visited:
                    continue
                neighbor_note = self.memories.get(neighbor_id)
                if neighbor_note is None:
                    continue
                ordered_notes.append(neighbor_note)
                visited.add(neighbor_id)
                frontier.append((neighbor_id, hop + 1))
                if len(ordered_notes) >= max_nodes:
                    break

        return self._dedupe_notes_by_id(ordered_notes)

    def compress_to_minimal_evidence(
        self,
        query: str,
        notes: List[MemoryNote],
        program: EvidenceProgram,
    ) -> List[MemoryNote]:
        deduped = self._dedupe_notes_by_id(notes)
        if len(deduped) <= program.max_notes:
            return deduped

        scored = [
            (note, self._score_note_for_program(query, note, program))
            for note in deduped
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        if program.program_type != ProgramType.MULTI_HOP:
            return [note for note, _ in scored[: program.max_notes]]

        selected: List[MemoryNote] = []
        selected_ids = set()
        while scored and len(selected) < program.max_notes:
            best_idx = 0
            best_value = None
            for idx, (note, base_score) in enumerate(scored):
                note_id = getattr(note, "id", None)
                link_bonus = 0.0
                if selected and note_id:
                    for chosen in selected:
                        chosen_links = set(getattr(chosen, "id_based_links", []) or [])
                        if note_id in chosen_links or getattr(chosen, "id", None) in (getattr(note, "id_based_links", []) or []):
                            link_bonus = 0.8
                            break
                value = base_score + link_bonus
                if best_value is None or value > best_value:
                    best_value = value
                    best_idx = idx
            note, _ = scored.pop(best_idx)
            note_id = getattr(note, "id", None)
            if note_id and note_id in selected_ids:
                continue
            selected.append(note)
            if note_id:
                selected_ids.add(note_id)

        return self._dedupe_notes_by_id(selected[: program.max_notes])

    def execute_evidence_program(
        self,
        program: EvidenceProgram,
        query: str,
        k: int = 10,
    ) -> RetrievalResult:
        trace = ExecutionTrace(metadata={"program": program.to_dict()})
        effective_k = max(1, min(int(k or program.k), max(program.k, k or program.k)))
        retrieval_engine = program.retrieval_engine

        if retrieval_engine == RetrievalEngine.DIRECT_HYBRID:
            notes = self.find_related_memories(query, k=effective_k)
            trace.add_step("retrieve_direct_hybrid", note_count=len(notes), k=effective_k)
            result = RetrievalResult(
                notes=notes,
                community_context="",
                communities=[],
                program_type=program.program_type.value,
                retrieval_engine=retrieval_engine.value,
            )
        elif retrieval_engine == RetrievalEngine.RRF_MULTI_CHANNEL:
            payload = self.retrieve_multi_channel_rrf(
                query=query,
                k=effective_k,
                program=program,
                channel_weights=program.channel_weights,
                enable_ppr=program.enable_ppr,
                ppr_max_hops=program.max_hops,
                ppr_damping=0.8,
                include_community_context=program.include_community_context,
                community_top_c=program.community_top_c,
            )
            trace.add_step(
                "retrieve_rrf",
                note_count=len(payload.get("notes", [])),
                enable_ppr=program.enable_ppr,
                max_hops=program.max_hops,
            )
            result = RetrievalResult.from_payload(
                payload,
                retrieval_engine=retrieval_engine.value,
                program_type=program.program_type.value,
            )
        elif retrieval_engine == RetrievalEngine.AGENTIC:
            payload = self.agentic_retrieve(
                query,
                k=effective_k,
                max_verify_per_subquery=min(max(effective_k, 8), 24),
                include_community_context=program.include_community_context,
                dense_weight=program.dense_weight,
            )
            trace.add_step("retrieve_agentic", max_verify_per_subquery=min(max(effective_k, 8), 24))
            result = RetrievalResult.from_payload(
                payload,
                retrieval_engine=retrieval_engine.value,
                program_type=program.program_type.value,
            )
        else:
            payload = self.find_related_memories_advanced(
                query,
                k=effective_k,
                include_community_context=program.include_community_context,
                max_hops=program.max_hops,
                alpha=program.fusion_alpha,
                dense_weight=program.dense_weight,
            )
            trace.add_step("retrieve_advanced", note_count=len(payload.get("notes", [])), max_hops=program.max_hops)
            result = RetrievalResult.from_payload(
                payload,
                retrieval_engine=RetrievalEngine.GRAPH_ADVANCED.value,
                program_type=program.program_type.value,
            )

        notes = self._dedupe_notes_by_id(result.notes)
        if program.program_type == ProgramType.MULTI_HOP:
            expanded_notes = self.expand_evidence_graph(notes, max_hops=program.max_hops)
            trace.add_step(
                "expand_graph",
                seed_count=len(notes),
                expanded_count=max(0, len(expanded_notes) - len(notes)),
                edge_hops=program.max_hops,
            )
            notes = self._dedupe_notes_by_id(expanded_notes)

        if program.include_community_context:
            community_payload = self.build_community_context_for_notes(notes, top_c=program.community_top_c)
        else:
            community_payload = {"community_context": "", "communities": []}

        result.notes = notes
        result.community_context = community_payload.get("community_context", "")
        result.communities = community_payload.get("communities", [])
        result.execution_trace = trace.to_dict()
        result.program_type = program.program_type.value
        result.retrieval_engine = retrieval_engine.value
        return result

    def save_graph(self, save_dir: str) -> None:
        """
        持久化 GraphRAG 状态：图边、社区摘要、社区 embeddings。

        参数:
            save_dir: 保存目录路径（会自动创建）
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 保存图边列表
        edges_path = Path(save_dir) / "graph_edges.json"
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(self.graph_edges, f, ensure_ascii=False, indent=2)

        # 保存社区摘要（不含 embedding）
        communities_path = Path(save_dir) / "communities.json"
        with open(communities_path, "w", encoding="utf-8") as f:
            json.dump(
                {cid: cs.to_dict() for cid, cs in self.communities.items()},
                f, ensure_ascii=False, indent=2
            )

        # 保存社区 embeddings
        if self.community_embeddings is not None:
            emb_path = Path(save_dir) / "community_embeddings.npy"
            np.save(str(emb_path), self.community_embeddings)

        # 保存 community_ids_list
        meta_path = Path(save_dir) / "graph_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "community_ids_list": self.community_ids_list,
                    "graph_cache_version": GRAPH_CACHE_VERSION,
                },
                f,
            )

        print(f"✅ GraphRAG 状态已保存到 {save_dir}")

    def load_graph(self, save_dir: str) -> None:
        """
        从磁盘加载 GraphRAG 状态。

        参数:
            save_dir: 保存目录路径
        """
        save_path = Path(save_dir)

        # 加载图边
        edges_path = save_path / "graph_edges.json"
        if edges_path.exists():
            with open(edges_path, "r", encoding="utf-8") as f:
                self.graph_edges = json.load(f)

        # 加载社区 embeddings
        emb_path = save_path / "community_embeddings.npy"
        if emb_path.exists():
            self.community_embeddings = np.load(str(emb_path))

        # 加载社区摘要
        communities_path = save_path / "communities.json"
        if communities_path.exists():
            with open(communities_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.communities = {}
            for cid, d in raw.items():
                cs = CommunitySummary.from_dict(d)
                # 恢复 embedding 到 CommunitySummary 对象中
                if self.community_embeddings is not None and "community_ids_list" in locals():
                     pass # handled below after loading meta
                self.communities[cid] = cs

        # 加载 meta 并恢复每个社区对象的 embedding 引用
        meta_path = save_path / "graph_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            graph_cache_version = meta.get("graph_cache_version", 0)
            if graph_cache_version != GRAPH_CACHE_VERSION:
                raise ValueError(
                    f"Graph cache version mismatch: expected {GRAPH_CACHE_VERSION}, got {graph_cache_version}"
                )
            self.community_ids_list = meta.get("community_ids_list", [])
            
            # 将 numpy array 形式的 embedding 分配回对应的 CommunitySummary
            if self.community_embeddings is not None and len(self.community_ids_list) == self.community_embeddings.shape[0]:
                for i, cid in enumerate(self.community_ids_list):
                    if cid in self.communities:
                        self.communities[cid].embedding = self.community_embeddings[i]

        print(
            f"✅ GraphRAG 状态已加载："
            f"{len(self.graph_edges)} 条边，{len(self.communities)} 个社区"
        )

    # ══════════════════════════════════════════════════════════════════════
    # ██  GraphRAG 方法结束
    # ══════════════════════════════════════════════════════════════════════
