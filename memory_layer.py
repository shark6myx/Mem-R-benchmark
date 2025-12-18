from ast import Str
from typing import Optional, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import requests
import json as json_lib
import time
import torch

def simple_tokenize(text):
    """
    简单的文本分词函数
    
    参数:
        text: 待分词的文本字符串
        
    返回:
        分词后的单词列表
    """
    return word_tokenize(text)

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

class HybridRetriever:
    """
    混合检索系统
    
    结合BM25和语义搜索的检索器
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.65):
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
            'model_name': 'all-MiniLM-L6-v2'  # 模型名称的默认值
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
            
        # 创建新实例
        retriever = cls(model_name=state['model_name'], alpha=state['alpha'])
        retriever.bm25 = state['bm25']
        retriever.corpus = state['corpus']
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
                    
                    # 只构建BM25索引
                    tokenized_docs = [doc.lower().split() for doc in all_docs]
                    retriever.bm25 = BM25Okapi(tokenized_docs)
                    
                    print(f"✓ 成功复用已有嵌入向量缓存，仅构建BM25索引")
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
            
        # 为BM25进行分词
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
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
        
        # 更新BM25
        if self.bm25 is None:
            # 第一个文档，初始化BM25
            tokenized_corpus = [simple_tokenize(document)]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            # 添加到现有BM25
            tokenized_doc = simple_tokenize(document)
            self.bm25.add_document(tokenized_doc)
        
        # 更新嵌入向量
        doc_embedding = self.model.encode([document], convert_to_tensor=True)
        if self.embeddings is None:
            self.embeddings = doc_embedding
        else:
            self.embeddings = torch.cat([self.embeddings, doc_embedding])
            
        return True
        
    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """
        使用混合评分检索文档
        
        参数:
            query: 查询文本
            k: 返回的文档数量
            
        返回:
            相关文档的索引列表
        """
        if not self.corpus:
            return []
            
        # 获取BM25得分
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 如果存在BM25得分，则进行归一化
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # 获取语义得分
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 组合得分
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        
        # 获取前k个索引
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return top_k_indices.tolist()

class SimpleEmbeddingRetriever:
    """
    简单嵌入检索系统
    
    仅使用文本嵌入向量的检索器
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化简单嵌入检索器
        
        参数:
            model_name: 使用的SentenceTransformer模型名称
        """
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # 文档内容到索引的映射
        
    def add_documents(self, documents: List[str]):
        """
        向检索器添加文档
        
        参数:
            documents: 文档文本列表
        """
        # 如果没有现有文档，则重置
        if not self.corpus:
            self.corpus = documents
            # print("documents", documents, len(documents))
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # 追加新文档
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """
        使用余弦相似度搜索相似文档
        
        参数:
            query: 查询文本
            k: 返回的结果数量
            
        返回:
            文档索引列表
        """
        if not self.corpus:
            return []
        # print("corpus", len(self.corpus), self.corpus)
        # 编码查询
        query_embedding = self.model.encode([query])[0]
        
        # 计算余弦相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # 获取前k个结果
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
            
        return top_k_indices
        
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
        
        # 保存其他属性
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """
        从磁盘加载检索器状态
        
        参数:
            retriever_cache_file: 检索器缓存文件路径
            retriever_cache_embeddings_file: 嵌入向量缓存文件路径
            
        返回:
            self，以便链式调用
        """
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # 加载嵌入向量
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # 加载其他属性
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """
        从内存中的记忆对象加载检索器状态
        
        参数:
            memories: 记忆字典
            model_name: 使用的模型名称
            
        返回:
            创建的检索器实例
        """
        # 为每个记忆创建结合内容和元数据的文档
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
            
        # 创建并初始化检索器
        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever

class AgenticMemorySystem:
    """
    智能记忆管理系统
    
    基于混合检索（BM25 + 语义搜索）的记忆管理系统
    """
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 retriever_alpha: float = 0.65):
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
        """
        self.memories = {}  # id -> MemoryNote
        self.retriever_alpha = retriever_alpha
        self.retriever = HybridRetriever(model_name, alpha=retriever_alpha)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
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
    
    def process_memory(self, note: MemoryNote) -> bool:
        """
        处理记忆笔记并返回演化标签
        
        参数:
            note: 要处理的记忆笔记对象
            
        返回:
            (should_evolve, note) 元组，should_evolve表示是否需要演化
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
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

    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """
        使用混合检索查找相关记忆
        
        参数:
            query: 查询文本
            k: 返回的相关记忆数量
            
        返回:
            (memory_str, indices) 元组，memory_str是格式化后的记忆字符串，indices是记忆索引列表
        """
        if not self.memories:
            return "",[]
            
        # 获取相关记忆的索引
        indices = self.retriever.retrieve(query, k)
        
        # 转换为记忆列表
        all_memories = list(self.memories.values())
        memory_str = ""
        # print("indices", indices)
        # print("all_memories", all_memories)
        for i in indices:
            memory_str += "memory index:" + str(i) + "\t talk start time:" + all_memories[i].timestamp + "\t memory content: " + all_memories[i].content + "\t memory context: " + all_memories[i].context + "\t memory keywords: " + str(all_memories[i].keywords) + "\t memory tags: " + str(all_memories[i].tags) + "\n"
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> List[MemoryNote]:
        """
        使用混合检索查找相关记忆（包含邻居记忆）
        
        参数:
            query: 查询文本
            k: 返回的相关记忆数量
            
        返回:
            格式化后的记忆字符串，包含邻居记忆的信息
        """
        if not self.memories:
            return []
            
        # 获取相关记忆的索引
        indices = self.retriever.retrieve(query, k)
        
        # 转换为记忆列表
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0
        for i in indices:
            memory_str +=  "talk start time:" + all_memories[i].timestamp + "memory content: " + all_memories[i].content + "memory context: " + all_memories[i].context + "memory keywords: " + str(all_memories[i].keywords) + "memory tags: " + str(all_memories[i].tags) + "\n"
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += "talk start time:" + all_memories[neighbor].timestamp + "memory content: " + all_memories[neighbor].content + "memory context: " + all_memories[neighbor].context + "memory keywords: " + str(all_memories[neighbor].keywords) + "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                if j >=k:
                    break
                j += 1
        return memory_str

def run_tests():
    """
    运行系统测试
    
    初始化记忆系统并添加测试记忆，然后查询相关记忆
    """
    print("Starting Memory System Tests...")
    
    # 使用OpenAI后端初始化记忆系统
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openai',
        llm_model='gpt-4o-mini'
    )
    
    print("\nAdding test memories...")
    
    # 添加测试记忆 - 只需提供内容
    memory_ids = []
    memory_ids.append(memory_system.add_note(
        "Neural networks are composed of layers of neurons that process information."
    ))
    
    memory_ids.append(memory_system.add_note(
        "Data preprocessing involves cleaning and transforming raw data for model training."
    ))
    
    print("\nQuerying for related memories...")
    query = MemoryNote(
        content="How do neural networks process data?",
        llm_controller=memory_system.llm_controller
    )
    
    related = memory_system.find_related_memories(query.content, k=2)
    print("related", related)
    print("\nResults:")
    for i, memory in enumerate(related, 1):
        print(f"\n{i}. Memory:")
        print(f"Content: {memory.content}")
        print(f"Category: {memory.category}")
        print(f"Keywords: {memory.keywords}")
        print(f"Tags: {memory.tags}")
        print(f"Context: {memory.context}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()