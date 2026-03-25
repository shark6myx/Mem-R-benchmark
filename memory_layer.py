from typing import Optional, Dict, List, Literal, Any, Union, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        初始化OpenAI控制器
        
        参数:
            model: 使用的模型名称，默认为"gpt-4"
            api_key: OpenAI API密钥，如果为None则使用默认密钥
        """
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = "sk-dxaeee451185337aec8ac82343fc46a73c5bf846cbf0wXWo"
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key,
                                 base_url="https://api.gptsapi.net/v1"
                                 )
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

    @retry(
        stop=stop_after_attempt(5),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=2, max=5),  # 重试间隔：指数退避，最小2秒，最大5秒
        retry=retry_if_exception_type((openai.APIConnectionError, openai.APIError))  # 仅对网络连接错误和API错误进行重试
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
                 backend: Literal["openai", "ollama", "sglang"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = "sk-dxaeee451185337aec8ac82343fc46a73c5bf846cbf0wXWo",
                 api_base: Optional[str] = None,
                 # sglang_host: str = "http://localhost",
                 # sglang_port: int = 30000
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
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            # 使用LiteLLM控制Ollama并支持JSON输出
            ollama_model = f"ollama/{model}" if not model.startswith("ollama/") else model
            self.llm = LiteLLMController(
                model=ollama_model, 
                api_base="http://localhost:11434", 
                api_key="EMPTY"
            )
        # elif backend == "sglang":
        #     # 直接调用SGLang API（性能更好，无需代理）
        #     self.llm = SGLangController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', or 'sglang'")

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
        # GraphRAG: community membership and stable id-based links
        self.community_id: Optional[str] = None
        self.id_based_links: List[str] = []  # stores note.id strings (not position indices)

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
        )


class HybridRetriever:
    """
    混合检索系统
    
    结合BM25和语义搜索的检索器
    """
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', alpha: float = 0.65):
        """
        初始化混合检索器
        
        参数:
            model_name: 使用的SentenceTransformer模型名称
            alpha: BM25和语义得分组合的权重（0 = 仅BM25，1 = 仅语义搜索）
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # 文档内容到索引的映射
        
    
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
            'alpha': self.alpha,
            'bm25': self.bm25,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': 'BAAI/bge-m3'  # 模型名称的默认值
        }
        
        # 尝试获取实际的模型名称（如果可能）
        try:
            state['model_name'] = self.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            pass
            
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
        
        # 使用默认值，防止缺少键的情况
        model_name = state.get('model_name', 'BAAI/bge-m3')
        alpha = state.get('alpha', 0.65)
            
        # 创建新实例
        retriever = cls(model_name=model_name, alpha=alpha)
        retriever.bm25 = state.get('bm25')
        retriever.corpus = state.get('corpus', [])
        retriever.document_ids = state.get('document_ids', {})
        
        # 如果存在嵌入向量文件，则从numpy文件加载
        embeddings_path = Path(retriever_cache_embeddings_file)
        if embeddings_path.exists():
            retriever.embeddings = np.load(embeddings_path)
            
        return retriever
    
    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str, alpha: float, 
                              embeddings_cache_file: str = None) -> 'HybridRetriever':
        """
        从内存中的记忆对象加载检索器状态
        
        参数:
            memories: 记忆字典
            model_name: 使用的模型名称
            alpha: BM25和语义得分组合的权重
            embeddings_cache_file: 可选的嵌入向量缓存文件路径，如果提供则复用已有嵌入向量
            
        返回:
            创建的检索器实例
        """
        all_docs = [", ".join(m.keywords) for m in memories.values()] #[m.content for m in memories.values()]
        retriever = cls(model_name, alpha)
        
        # 如果提供了嵌入向量缓存文件且文件存在，尝试加载
        if embeddings_cache_file and Path(embeddings_cache_file).exists():
            try:
                cached_embeddings = np.load(embeddings_cache_file)
                # 验证嵌入向量数量是否匹配
                if len(cached_embeddings) == len(all_docs):
                    retriever.corpus = all_docs
                    retriever.embeddings = cached_embeddings
                    retriever.document_ids = {doc: idx for idx, doc in enumerate(all_docs)}
                    
                    # 根据升级计划弃用 BM25
                    # tokenized_docs = [doc.lower().split() for doc in all_docs]
                    # retriever.bm25 = BM25Okapi(tokenized_docs)
                    
                    print(f"✓ 成功复用已有嵌入向量缓存")
                    return retriever
                else:
                    print(f"⚠ 嵌入向量数量不匹配（缓存: {len(cached_embeddings)}, 文档: {len(all_docs)}），将重新生成")
            except Exception as e:
                print(f"⚠ 加载嵌入向量缓存失败: {e}，将重新生成")
        
        # 如果没有缓存或加载失败，使用原始方法
        retriever.add_documents(all_docs)
        return retriever
    
    def add_documents(self, documents: List[str]) -> bool:
        """
        一次性添加文档到BM25和语义索引中
        
        参数:
            documents: 文档文本列表
            
        返回:
            True表示成功添加
        """
        if not documents:
            return
            
        # 为BM25进行分词 (弃用)
        # tokenized_docs = [doc.lower().split() for doc in documents]
        # self.bm25 = BM25Okapi(tokenized_docs)
        
        # 创建嵌入向量
        self.embeddings = self.model.encode(documents)
        self.corpus = documents
        doc_idx = 0
        for document in documents:
            self.document_ids[document] = doc_idx
            doc_idx += 1

        return True

    def add_document(self, document: str) -> bool:
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
            
        # 添加到语料库并获取索引
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx
        
        # 更新BM25 (弃用)
        # if self.bm25 is None:
        #     tokenized_corpus = [simple_tokenize(document)]
        #     self.bm25 = BM25Okapi(tokenized_corpus)
        # else:
        #     tokenized_doc = simple_tokenize(document)
        #     self.bm25.add_document(tokenized_doc)
        
        # 更新嵌入向量
        # 修复：使用 numpy 进行拼接，避免 tensor 类型不匹配错误
        doc_embedding = self.model.encode([document])
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embedding])
            
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
        
        # BM25 已弃用，始终走纯 Dense 语义路径
        if self.embeddings is None:
            return ([] if not return_scores else ([], []))
        query_embedding = self.model.encode([query])[0]
        hybrid_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
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

        embeddings_arr = np.array(self.embeddings)
        norm_q = np.linalg.norm(query_embedding)
        norm_e = np.linalg.norm(embeddings_arr, axis=1)
        norm_e[norm_e == 0] = 1e-10
        if norm_q == 0:
            norm_q = 1e-10

        return np.dot(embeddings_arr, query_embedding) / (norm_e * norm_q)



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
    
    基于混合检索（BM25 + 语义搜索）的记忆管理系统
    """
    def __init__(self, 
                 model_name: str = 'BAAI/bge-m3',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 retriever_alpha: float = 0.65,
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
            retriever_alpha: 混合检索器中BM25和语义得分的权重（0 = 仅BM25，1 = 仅语义搜索）
            use_reranking: 是否启用重排序（使用CrossEncoder提升检索精度）
        """
        self.memories = {}  # id -> MemoryNote
        self.retriever_alpha = retriever_alpha
        self.retriever = HybridRetriever(model_name, alpha=retriever_alpha)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
        # 重排序相关
        self.use_reranking = use_reranking
        self.reranker = None  # 延迟加载
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
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
        self.edge_semantic_threshold: float = 0.70
        # 关键词共现边阈值（Jaccard）
        self.edge_jaccard_threshold: float = 0.20
        # ────────────────────────────────────────────────────────────

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
        doc_text = "content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)
        self.retriever.add_document(doc_text)
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        # ── GraphRAG：构建图边 + 触发社区重聚类 ─────────────────────
        self._build_edges_for_new_note(note)
        self.note_total_count += 1
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
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            # 回退：使用类初始化时的模型名称
            model_name = 'all-MiniLM-L6-v2'
        
        self.retriever = HybridRetriever(model_name, alpha=self.retriever_alpha)
        
        # 收集所有记忆文档并一次性添加
        all_docs = []
        for memory in self.memories.values():
            # 将记忆元数据合并为单个可搜索文档
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            doc_text = memory.content + " , " + metadata_text
            all_docs.append(doc_text)
        
        # 一次性添加所有文档（HybridRetriever的add_documents会重新初始化索引）
        if all_docs:
            self.retriever.add_documents(all_docs)

        # ── GraphRAG：同步重建社区 ──────────────────────────────────────
        if len(self.memories) >= 5:
            self.rebuild_communities()
        # ────────────────────────────────────────────────────────────────
    
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
        all_memories = list(self.memories.values())
        neighbor_memory = ""
        for i in indices:
            neighbor_memory += "memory index:" + str(i) + "\t talk start time:" + all_memories[i].timestamp + "\t memory content: " + all_memories[i].content + "\t memory context: " + all_memories[i].context + "\t memory keywords: " + str(all_memories[i].keywords) + "\t memory tags: " + str(all_memories[i].tags) + "\n"

        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
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
                            "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
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
                    suggest_connections = response_json["suggested_connections"]
                    new_tags = response_json["tags_to_update"]
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    # 如果小语言模型输出的数量少于邻居数量，则按顺序使用新标签和上下文
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        # 查找某个记忆
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # 向记忆添加标签
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve,note

    def _get_reranker(self):
        """
        延迟加载重排序器（CrossEncoder）
        
        返回:
            CrossEncoder实例，如果不可用则返回None
        """
        if self.reranker is None and self.use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                # 使用轻量级的交叉编码器模型
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✓ 重排序器（CrossEncoder）已加载")
            except ImportError:
                print("⚠ Warning: CrossEncoder not available, reranking disabled")
                self.use_reranking = False
            except Exception as e:
                print(f"⚠ Warning: Failed to load reranker: {e}, reranking disabled")
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
        
        all_memories = list(self.memories.values())
        
        # 构建查询-文档对
        pairs = []
        for idx in retrieved_indices:
            mem = all_memories[idx]
            # 构建文档文本（用于重排序）
            # 优化顺序：关键词 + 标签 + 内容 + 上下文（确保关键元数据不被截断）
            doc_text = f"{', '.join(mem.keywords)} , {', '.join(mem.tags)} , {mem.content} , {mem.context}"
            pairs.append([query, doc_text])
        
        # 使用交叉编码器评分（更精确但更慢）
        try:
            scores = reranker.predict(pairs)
            # 按分数排序（分数越高越相关）
            ranked_indices = [retrieved_indices[i] for i in np.argsort(scores)[::-1]]
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
        all_memories = list(self.memories.values())
        return [all_memories[i] for i in indices]



    def agentic_retrieve(self, query: str, k: int = 5) -> str:
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
        
        # 3. 定向检索与反思过滤
        for sub_q in sub_queries:
            # 加上核心意图，防止偏离
            context_query = f"Intent: {core_intent} Sub-query: {sub_q}"
            
            # 使用升级版的超能漫游检索
            advanced_result = self.find_related_memories_advanced(context_query, k=k)
            raw_memories = advanced_result.get("notes", [])
            
            for mem in raw_memories:
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
            
        # 5. 验证假设文档的逻辑闭环
        closure_check = verifier.verify_logical_closure(query, all_relevant_contexts)
        
        # 如果逻辑非闭环，尝试用缺失信息进行二次补充检索 (Anti-drift Follow-up)
        if not closure_check.get("is_closed_loop", True):
            missing_info = closure_check.get("missing_information", "")
            print(f"↻ Logical loop incomplete. Missing: {missing_info}. Triggering follow-up retrieval...")
            followup_query = f"Intent: {core_intent} Missing: {missing_info}"
            followup_result = self.find_related_memories_advanced(followup_query, k=2)
            followup_memories = followup_result.get("notes", [])
            
            for mem in followup_memories:
                doc_text = f"Content: {mem.content} Keywords: {mem.keywords}"
                if doc_text not in seen_documents:
                    seen_documents.add(doc_text)
                    all_relevant_contexts.append(
                        f"[Follow-up Evidence for '{missing_info}'] {doc_text}\n"
                        f"Reflection: <reflection> Retrieved to resolve logical gap </reflection>"
                    )
            
        final_context = "\n---\n".join(all_relevant_contexts)
        return final_context

    # ══════════════════════════════════════════════════════════════════════
    # ██  GraphRAG 社区聚类方法（新增）
    # ══════════════════════════════════════════════════════════════════════

    def _build_edges_for_new_note(self, note: MemoryNote) -> None:
        """
        为新加入的 note 自动构建三类图边：
          1. semantic  : 语义 cosine 相似度 ≥ edge_semantic_threshold
          2. temporal  : timestamp 前缀相同（同一 session）且连续
          3. entity_shared : 关键词 Jaccard 相似度 ≥ edge_jaccard_threshold

        构建完成后，同时更新 note.id_based_links（双向）。
        """
        if not self.memories:
            return

        # 获取新 note 的嵌入向量（直接用检索器已经 encode 过的最后一条）
        # 由于 add_note 在调用此函数前已经 retriever.add_document，
        # 最后一个嵌入就是新 note 的向量
        if self.retriever.embeddings is None or len(self.retriever.embeddings) == 0:
            return

        new_emb = self.retriever.embeddings[-1].reshape(1, -1)   # shape (1, D)

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
                prev_doc_key = (
                    "content:" + prev_note.content
                    + " context:" + prev_note.context
                    + " keywords: " + ", ".join(prev_note.keywords)
                    + " tags: " + ", ".join(prev_note.tags)
                )
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

        # ── 2. 时序边（同 session 内相邻两条） ──────────────────────────
        # timestamp 格式为 YYYYMMDDHHmm，取前8位作为 session key（同一天）
        new_session = note.timestamp[:8] if note.timestamp else ""
        for prev_note in existing_notes[-5:]:   # 只看最近5条，够用且高效
            prev_session = prev_note.timestamp[:8] if prev_note.timestamp else ""
            if new_session and prev_session and new_session == prev_session:
                self.graph_edges.append({
                    "src_id": prev_note.id,
                    "tgt_id": note.id,
                    "weight": 0.6,
                    "type": "temporal"
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

        # ── 4. LLM evolution 产生的 strengthen links → id_based_links ──
        # process_memory 写入了 note.links（位置索引），这里同步转 id
        # （process_memory 在此函数调用前已执行，self.memories 已含新 note）
        all_ids = list(self.memories.keys())
        for pos_idx in note.links:
            if isinstance(pos_idx, int) and 0 <= pos_idx < len(all_ids):
                linked_id = all_ids[pos_idx]
                if linked_id != note.id and linked_id not in note.id_based_links:
                    note.id_based_links.append(linked_id)
                    self.graph_edges.append({
                        "src_id": note.id,
                        "tgt_id": linked_id,
                        "weight": 0.8,
                        "type": "llm_inferred"
                    })

    def _build_networkx_graph(self):
        """
        将 self.graph_edges 转成 networkx 无向加权图，供社区检测使用。

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

        # 添加边（同向去重，取最大权重）
        edge_map: Dict[Tuple[str, str], float] = {}
        for e in self.graph_edges:
            src, tgt, w = e["src_id"], e["tgt_id"], e["weight"]
            if src not in self.memories or tgt not in self.memories:
                continue
            key = (min(src, tgt), max(src, tgt))
            edge_map[key] = max(edge_map.get(key, 0.0), w)

        for (src, tgt), w in edge_map.items():
            G.add_edge(src, tgt, weight=w)

        return G

    def _run_community_detection(self, G) -> Dict[str, int]:
        """
        在图 G 上运行 Louvain 社区检测。

        参数:
            G: networkx.Graph 对象

        返回:
            {note_id: community_int} 映射字典
        """
        try:
            import community as community_louvain
        except ImportError:
            raise ImportError(
                "python-louvain 未安装，请运行: pip install python-louvain"
            )

        if len(G.nodes) == 0:
            return {}

        # 孤立节点（无边）也可被 Louvain 分配到单独社区
        # 注意：python-louvain 的 best_partition 不支持 random_state 参数
        partition = community_louvain.best_partition(G, weight="weight")
        return partition   # {node_id_str: community_int}

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
        notes_text = ""
        for m in members[:20]:   # 最多取前20条避免超 context
            notes_text += (
                f"[{m.timestamp}] {m.content} "
                f"(keywords: {', '.join(m.keywords[:5])})\n"
            )

        prompt = f"""你是一个记忆聚类分析专家。以下是属于同一主题聚类的 {len(members)} 条记忆片段：

{notes_text}

请综合分析这组记忆，生成：
1. title：一句话概括核心主题（≤20字）
2. summary：对这组记忆的综合描述（≤200字），涵盖主要事件、实体、时间线
3. keywords：5-10个代表性关键词列表

以JSON格式返回：{{"title": "...", "summary": "...", "keywords": [...]}}"""

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "community_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title":    {"type": "string"},
                        "summary":  {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
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

        # 对摘要文本生成 embedding
        embedding = None
        if summary:
            try:
                embedding = self.retriever.model.encode([summary])[0]
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

    def rebuild_communities(self) -> None:
        """
        GraphRAG 社区重建：
          1. 构建 networkx 图
          2. 运行 Louvain 社区检测
          3. 为每个社区生成 LLM 摘要
          4. 更新 self.communities 和 community_embeddings

        会自动更新每条 note 的 community_id 字段。
        """
        if len(self.memories) < 2:
            return

        print(f"🔄 GraphRAG: 重建社区（共 {len(self.memories)} 条记忆）...")

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

        # Step 3: 按社区 id 分组
        community_groups: Dict[int, List[str]] = {}
        for note_id, comm_int in partition.items():
            community_groups.setdefault(comm_int, []).append(note_id)

        # Step 4: 为每个社区生成摘要
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
            new_communities[community_id] = cs

            if cs.embedding is not None:
                all_embeddings.append(cs.embedding)
                cid_order.append(community_id)

        # Step 5: 更新类属性
        self.communities = new_communities
        self.community_ids_list = cid_order
        if all_embeddings:
            self.community_embeddings = np.stack(all_embeddings, axis=0)
        else:
            self.community_embeddings = None

        print(
            f"✅ GraphRAG: 社区重建完成，共 {len(new_communities)} 个社区："
        )
        for cid, cs in new_communities.items():
            print(f"   [{cid}] {cs.title} ({len(cs.member_note_ids)} 条记忆)")

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
        all_memories = list(self.memories.values())
        related_notes = [all_memories[i] for i in indices if i < len(all_memories)]

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
            community_context += (
                f"【{cs.title}】{cs.summary} "
                f"(涉及关键词: {', '.join(cs.keywords[:5])})\n"
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
            return hyde_doc.strip()
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

    def _ppr_walk(self, seed_ids: list, damping: float = 0.8, max_hops: int = 2) -> dict:
        """
        [PPR Multi-hop] 个性化页面排名（多跳漫游）
        从种子节点出发，利用 semantic, temporal, Jaccard 图边扩展隐藏线索。
        """
        from collections import defaultdict
        scores = defaultdict(float)
        
        # 种子节点初始分数
        for sid in seed_ids:
            scores[sid] = 1.0
            
        # 预加载无向图邻接表
        adj = defaultdict(lambda: defaultdict(float))
        for edge in self.graph_edges:
            src, tgt, w = edge.get("src_id"), edge.get("tgt_id"), edge.get("weight", 0.1)
            # 无向图，保存最大权重
            adj[src][tgt] = max(adj[src][tgt], w)
            adj[tgt][src] = max(adj[tgt][src], w)
            
        current_frontier = set(seed_ids)
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            for node in current_frontier:
                current_score = scores[node]
                for neighbor, weight in adj[node].items():
                    # 传播得分: 节点现有得分 * 边权重 * 衰减率
                    propagated = current_score * float(weight) * damping
                    if propagated > scores[neighbor]:
                        scores[neighbor] = propagated
                        next_frontier.add(neighbor)
            current_frontier = next_frontier
            
        return dict(scores)

    def find_related_memories_advanced(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        终极检索演进管道：HyDE + Community Filter + BGE-M3 + PPR Walk
        返回含有丰富线索组合的综合证据字典。
        完全兼容原有的返回结构。
        """
        all_memories_list = list(self.memories.values())
        if not all_memories_list:
             return {"notes": [], "community_context": "", "communities": []}
             
        # 1. HyDE
        hyde_doc = self._generate_hyde_query(query)
        
        # 2. Vectorize
        query_emb = self.retriever.model.encode([hyde_doc])
        
        # 3. Community Filter
        allowed_ids = self._community_filter(query_emb, top_c=2)
        
        # 4. Dense 精搜种子节点
        scores = self.retriever.get_scores_by_emb(query_emb[0])
        
        if len(scores) == 0:
            return {"notes": [], "community_context": "", "communities": []}
            
        seed_scores = []
        for i, doc_str in enumerate(self.retriever.corpus):
            # 确保下标在字典范围内
            if i < len(all_memories_list):
                mem = all_memories_list[i]
                if allowed_ids is None or mem.id in allowed_ids:
                    seed_scores.append((mem.id, scores[i]))
                    
        # 提取 Top-K 种子
        seed_scores.sort(key=lambda x: x[1], reverse=True)
        top_seeds = [t[0] for t in seed_scores[:k]]
        
        # 5. PPR 多跳
        if top_seeds:
            ppr_results = self._ppr_walk(top_seeds, damping=0.8, max_hops=2)
        else:
            ppr_results = {}
            
        # 6. 合并收网 (过滤掉得分过低的节点)
        final_list = []
        for mid, sc in ppr_results.items():
            if mid in self.memories and sc > 0.1:
                final_list.append((self.memories[mid], sc))
                
        final_list.sort(key=lambda x: x[1], reverse=True)
        related_notes = [x[0] for x in final_list[:k*2]] # 留多点证据给后续的Reflection验证
        
        # 如果依然不够，回退填充
        if len(related_notes) < k and seed_scores:
             existing_ids = set(n.id for n in related_notes)
             for t in seed_scores:
                 if t[0] not in existing_ids and t[0] in self.memories:
                     related_notes.append(self.memories[t[0]])
                 if len(related_notes) >= k*2:
                     break
                     
        # 7. 拼接社区语境 (Community Context)
        involved_community_ids = set()
        for note in related_notes:
            if note.community_id and note.community_id in self.communities:
                involved_community_ids.add(note.community_id)

        involved_communities = [self.communities[cid] for cid in involved_community_ids]

        community_context = ""
        for cs in involved_communities:
            community_context += (
                f"【{cs.title}】{cs.summary} "
                f"(涉及关键词: {', '.join(cs.keywords[:5])})\n"
            )

        return {
            "notes": related_notes,
            "community_context": community_context.strip(),
            "communities": involved_communities,
        }

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
            json.dump({"community_ids_list": self.community_ids_list}, f)

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

        # 加载社区摘要
        communities_path = save_path / "communities.json"
        if communities_path.exists():
            with open(communities_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.communities = {
                cid: CommunitySummary.from_dict(d) for cid, d in raw.items()
            }

        # 加载社区 embeddings
        emb_path = save_path / "community_embeddings.npy"
        if emb_path.exists():
            self.community_embeddings = np.load(str(emb_path))

        # 加载 meta
        meta_path = save_path / "graph_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.community_ids_list = meta.get("community_ids_list", [])

        print(
            f"✅ GraphRAG 状态已加载："
            f"{len(self.graph_edges)} 条边，{len(self.communities)} 个社区"
        )

    # ══════════════════════════════════════════════════════════════════════
    # ██  GraphRAG 方法结束
    # ══════════════════════════════════════════════════════════════════════