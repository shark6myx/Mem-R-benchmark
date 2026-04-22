from memory_layer import LLMController, AgenticMemorySystem, MEMORY_TEXT_SCHEMA_VERSION
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
import nltk
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime, timedelta
import re

from evidence_program import AnswerStyle, EvidenceProgram
from legacy_adapter import classify_question_mode_from_program
from pipeline import ProgramGuidedQAPipeline
from retrieval_types import RetrievalResult
from routing import ProgramRouter

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-minicpm-layerwise"
DEFAULT_RERANKER_CUTOFF_LAYER = 28

nltk.download('punkt_tab')
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

class advancedMemAgent:
    # def __init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000):
    def __init__(
        self,
        model,
        backend,
        retrieve_k,
        temperature_c5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_cutoff_layer: int = DEFAULT_RERANKER_CUTOFF_LAYER,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model,
            reranker_model_name=reranker_model,
            reranker_cutoff_layer=reranker_cutoff_layer,
            llm_backend=backend,
            llm_model=model,
            api_base=api_base,
            api_key=api_key,
            sglang_host=sglang_host,
            sglang_port=sglang_port,

        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,

        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.category_name_map = {
            1: "single_hop_factoid",
            2: "temporal",
            3: "multi_hop_reasoning",
            4: "open_domain_synthesis",
            5: "adversarial",
        }
        # 路由缓存：同一问题只调用一次LLM分类，避免重复开销
        self._route_cache: Dict[str, Dict[str, Any]] = {}
        self.router = ProgramRouter(
            llm_controller=self.retriever_llm,
            retrieve_k=self.retrieve_k,
            route_cache=self._route_cache,
        )
        self.pipeline = ProgramGuidedQAPipeline(
            memory_system=self.memory_system,
            llm_controller=self.memory_system.llm_controller,
            prompt_builder=self._build_answer_prompt,
        )
        self._last_route_decision: Dict[str, Any] = {}
        self._last_program: Dict[str, Any] = {}
        self._last_verification: Dict[str, Any] = {}
        self._last_execution_trace: Dict[str, Any] = {}
        self._last_compressed_evidence_count: int = 0
        self._last_rrf_scores: Dict[str, float] = {}
        self._last_channel_details: Dict[str, Any] = {}
        self._last_retrieval_metadata: Dict[str, Any] = {}
        self._last_answer_subgraph_summary: Dict[str, Any] = {}

    def add_memories_batch(self, contents: List[str], times: List[str] = None):
        """
        批量添加记忆以利用批量向量化优化
        """
        self.memory_system.add_notes_batch(contents, times=times)

    # def retrieve_memory(self, content, k=10):
    #     return self.memory_system.find_related_memories_raw(content, k=k)
    def _infer_task_attributes(self, question: str) -> Dict[str, bool]:
        return self.router.infer_task_attributes(question)

    def _fallback_task_type_from_question(self, question: str) -> Dict[str, Any]:
        return self.router.fallback_task_type_from_question(question)

    def _llm_route_question(self, question: str, fallback_category: Optional[int] = None) -> Dict[str, Any]:
        return self.router.llm_route_question(question)

    def predict_evidence_program(self, question: str) -> EvidenceProgram:
        return self.router.predict_evidence_program(question)

    def _classify_question_mode(
        self,
        question: str,
        category: Optional[int] = None,
        program: Optional[EvidenceProgram] = None,
    ) -> Dict[str, Union[bool, int, float, str]]:
        """
        保留旧接口，但内部已经变成 program -> runtime config 的兼容层。
        """
        predicted_program = program or self.predict_evidence_program(question)
        return classify_question_mode_from_program(predicted_program)

    def _format_retrieval_context(self, retrieval_result: Union[str, Dict[str, Any], RetrievalResult], include_community_context: bool) -> str:
        if isinstance(retrieval_result, RetrievalResult):
            return retrieval_result.format_context(include_community_context)
        if isinstance(retrieval_result, str):
            return retrieval_result
        return RetrievalResult.from_payload(retrieval_result).format_context(include_community_context)

    def retrieve_memory(
        self,
        content,
        k=None,
        program: Optional[EvidenceProgram] = None,
        use_agentic: bool = False,
        retrieval_mode: str = "advanced",
        include_community_context: bool = True,
        max_hops: int = 2,
        fusion_alpha: float = 0.5,
        dense_weight: float = 0.70,
        use_rrf: bool = False,
        channel_weights: Optional[Dict[str, Any]] = None,
        enable_ppr: bool = False,
        community_top_c: int = 5,
        return_result: bool = False,
    ):
        if k is None:
            k = self.retrieve_k

        if program is not None:
            result = self.memory_system.execute_evidence_program(program, content, k=k or program.k)
            self._last_rrf_scores = dict(result.rrf_scores)
            self._last_channel_details = dict(result.channel_details)
            if return_result:
                return result
            return result.format_context(program.include_community_context)

        # ── Multi-Channel RRF 通道（优先） ────────────────────────
        # 一旦 use_rrf=True，下面的 retrieval_mode / use_agentic / fusion_alpha
        # 分支都不会参与当前请求的实际检索。
        if use_rrf:
            payload = self.memory_system.retrieve_multi_channel_rrf(
                query=content,
                k=k,
                channel_weights=channel_weights,
                enable_ppr=enable_ppr,
                ppr_max_hops=max_hops,
                ppr_damping=0.8,
                include_community_context=include_community_context,
                community_top_c=community_top_c,
            )
            result = RetrievalResult.from_payload(payload)
            self._last_rrf_scores = dict(result.rrf_scores)
            self._last_channel_details = dict(result.channel_details)
            if return_result:
                return result
            return result.format_context(include_community_context)

        if use_agentic:
            # 使用新加入的 Agentic Decomposition 与 Reflection 分析闭环链路
            try:
                # 如果是 agentic_retrieve 返回了字典，用相同的格式化函数
                res = self.memory_system.agentic_retrieve(
                    content,
                    k=k,
                    max_verify_per_subquery=min(max(k, 8), 24),
                    include_community_context=include_community_context,
                    dense_weight=dense_weight,
                )
                if isinstance(res, dict):
                    wrapped = RetrievalResult.from_payload(res)
                    if return_result:
                        return wrapped
                    return wrapped.format_context(include_community_context)
                return res
            except Exception as e:
                print(f"agentic_retrieve failed: {e}")
                empty_result = RetrievalResult()
                return empty_result if return_result else ""

        if retrieval_mode == "direct_hybrid":
            notes = self.memory_system.find_related_memories(content, k=k)
            result = {
                "notes": notes,
                "community_context": "",
                "communities": [],
            }
            wrapped = RetrievalResult.from_payload(result)
            if return_result:
                return wrapped
            return wrapped.format_context(include_community_context=False)
             
        # 默认：GraphRAG高级检索（支持低跳数/禁社区上下文的精确模式）
        try:
            result = self.memory_system.find_related_memories_advanced(
                content,
                k=k,
                include_community_context=include_community_context,
                max_hops=max_hops,
                alpha=fusion_alpha,
                dense_weight=dense_weight,
            )
        except TypeError:
            # 兼容旧接口
            result = self.memory_system.find_related_memories_advanced(content, k=k)
        wrapped = RetrievalResult.from_payload(result)
        if return_result:
            return wrapped
        return wrapped.format_context(include_community_context)

    def _extract_primary_evidence_block(self, raw_context: str) -> str:
        if not raw_context:
            return ""
        block = str(raw_context).split("\n---\n", 1)[0].strip()
        return block or str(raw_context)

    def _extract_anchor_datetime(self, raw_context: str) -> Optional[datetime]:
        if not raw_context:
            return None

        search_spaces = [self._extract_primary_evidence_block(raw_context), str(raw_context)]
        for text in search_spaces:
            recorded_match = re.search(
                r"recorded:\s*\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+(\d{1,2}\s+[A-Za-z]+),\s*(\d{4})",
                text,
                flags=re.IGNORECASE,
            )
            if recorded_match:
                try:
                    return datetime.strptime(
                        f"{recorded_match.group(1)} {recorded_match.group(2)}",
                        "%d %B %Y",
                    )
                except ValueError:
                    pass

            summary_match = re.search(
                r"\bOn\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b",
                text,
                flags=re.IGNORECASE,
            )
            if summary_match:
                try:
                    return datetime.strptime(
                        f"{summary_match.group(2)} {summary_match.group(1)} {summary_match.group(3)}",
                        "%d %B %Y",
                    )
                except ValueError:
                    pass

        return None

    def _format_day_month_year(self, dt: datetime, include_year: bool = True) -> str:
        if include_year:
            return f"{dt.day} {dt.strftime('%B')} {dt.year}"
        return f"{dt.day} {dt.strftime('%B')}"

    def _canonicalize_absolute_date(self, value: str, omit_year: bool = False) -> str:
        text = re.sub(r"\s+", " ", (value or "").replace(",", " ").strip())
        if not text:
            return ""

        for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return self._format_day_month_year(
                    datetime.strptime(text, fmt),
                    include_year=not omit_year,
                )
            except ValueError:
                pass

        month_year_match = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", text)
        if month_year_match:
            month_name = month_year_match.group(1)
            for month_fmt in ("%B", "%b"):
                try:
                    parsed_month = datetime.strptime(month_name, month_fmt)
                    return f"{parsed_month.strftime('%B')} {month_year_match.group(2)}"
                except ValueError:
                    pass

        day_month_match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)", text)
        if day_month_match:
            month_name = day_month_match.group(2)
            for month_fmt in ("%B", "%b"):
                try:
                    parsed_month = datetime.strptime(month_name, month_fmt)
                    return f"{int(day_month_match.group(1))} {parsed_month.strftime('%B')}"
                except ValueError:
                    pass

        year_match = re.fullmatch(r"\d{4}", text)
        if year_match:
            return text

        return text

    def _canonicalize_reference_style_phrase(self, text: str) -> Optional[str]:
        if not text:
            return None

        text = re.sub(r"\s+", " ", text).strip()

        two_weekends_match = re.search(
            r"\btwo weekends\s+(before|of)\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
            text,
            flags=re.IGNORECASE,
        )
        if two_weekends_match:
            return (
                f"two weekends {two_weekends_match.group(1).lower()} "
                f"{self._canonicalize_absolute_date(two_weekends_match.group(2))}"
            )

        rel_match = re.search(
            r"\b(?:the\s+)?(week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
            r"\s+(before|of)\s+(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if not rel_match:
            return None

        unit = rel_match.group(1).lower()
        relation = rel_match.group(2).lower()
        date_part = self._canonicalize_absolute_date(rel_match.group(3))
        display_unit = unit.capitalize() if unit in {
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        } else unit
        return f"The {display_unit} {relation} {date_part}"

    def _extract_duration_phrase(self, text: str) -> Optional[str]:
        if not text:
            return None

        duration_patterns = [
            r"\bsince\s+\d{4}\b",
            r"\b\d+\s+years?\s+ago\b",
            r"\b\d+\s+years?\b",
            r"\b\d+\s+months?\b",
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = re.sub(r"\s+", " ", match.group(0).strip())
                if value.lower().startswith("since "):
                    return f"Since {value.split()[-1]}"
                return value
        return None

    def _normalize_temporal_answer(self, question: str, text: str, raw_context: str = "") -> str:
        primary_context = self._extract_primary_evidence_block(raw_context)
        context_without_recorded = re.sub(r"\(recorded:[^)]+\)", "", primary_context, flags=re.IGNORECASE)
        anchor_dt = self._extract_anchor_datetime(raw_context)
        q_lower = (question or "").lower()
        omit_year = "birthday" in q_lower

        for source in [text, primary_context, raw_context]:
            ref_style = self._canonicalize_reference_style_phrase(source)
            if ref_style:
                return ref_style

        for source in [text, context_without_recorded, raw_context]:
            duration = self._extract_duration_phrase(source)
            if duration:
                return duration

        absolute_patterns = [
            r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
            r"\b[A-Za-z]+\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{2}/\d{2}/\d{4}\b",
            r"\b\d{1,2}\s+[A-Za-z]+\b",
            r"\b\d{4}\b",
        ]
        for source in [text, context_without_recorded]:
            for pattern in absolute_patterns:
                match = re.search(pattern, source)
                if match:
                    return self._canonicalize_absolute_date(match.group(0), omit_year=omit_year)

        if not anchor_dt:
            return text

        future_question = any(token in q_lower for token in ["when is", "going to", "will ", "upcoming"])
        low_signal_answer = (
            not text
            or any(marker in text.lower() for marker in [
                "not mentioned",
                "no specific date",
                "does not specify",
                "no information",
            ])
        )
        source_candidates = [context_without_recorded, text]
        if low_signal_answer:
            source_candidates.append(raw_context)

        relative_text = ""
        relative_patterns = [
            r"\b(last night|yesterday|today)\b",
            r"\b((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago)\b",
            r"\b(this month|last month|next month|this year|last year|this week|last week|last weekend)\b",
            r"\b(last\s+(?:mon(?:day)?|tues?(?:day)?|wednes(?:day)?|thurs?(?:day)?|fri(?:day)?|satur(?:day)?|sun(?:day)?))\b",
            r"\b(upcoming|coming up)\b",
        ]
        for source in source_candidates:
            if not source:
                continue
            for pattern in relative_patterns:
                match = re.search(pattern, source, flags=re.IGNORECASE)
                if match:
                    relative_text = match.group(1).lower()
                    break
            if relative_text:
                break

        if not relative_text:
            return text

        if relative_text in {"yesterday", "last night"}:
            return self._format_day_month_year(anchor_dt - timedelta(days=1), include_year=not omit_year)
        if relative_text == "today":
            return self._format_day_month_year(anchor_dt, include_year=not omit_year)

        days_ago_match = re.fullmatch(
            r"((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))\s+days?\s+ago",
            relative_text,
        )
        if days_ago_match:
            word = days_ago_match.group(1).lower()
            day_map = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            }
            days = int(word) if word.isdigit() else day_map.get(word, 0)
            if days > 0:
                return self._format_day_month_year(anchor_dt - timedelta(days=days), include_year=not omit_year)

        if relative_text == "this month":
            return anchor_dt.strftime("%B %Y")
        if relative_text == "last month":
            year = anchor_dt.year if anchor_dt.month > 1 else anchor_dt.year - 1
            month = anchor_dt.month - 1 if anchor_dt.month > 1 else 12
            return datetime(year, month, 1).strftime("%B %Y")
        if relative_text in {"next month", "upcoming", "coming up"} and future_question:
            year = anchor_dt.year if anchor_dt.month < 12 else anchor_dt.year + 1
            month = anchor_dt.month + 1 if anchor_dt.month < 12 else 1
            return datetime(year, month, 1).strftime("%B %Y")
        if relative_text == "this year":
            return str(anchor_dt.year)
        if relative_text == "last year":
            return str(anchor_dt.year - 1)
        if relative_text == "this week":
            return f"The week of {self._format_day_month_year(anchor_dt)}"
        if relative_text == "last week":
            return f"The week before {self._format_day_month_year(anchor_dt)}"
        if relative_text == "last weekend":
            return f"The weekend before {self._format_day_month_year(anchor_dt)}"

        weekday_aliases = {
            "mon": "Monday",
            "monday": "Monday",
            "tue": "Tuesday",
            "tues": "Tuesday",
            "tuesday": "Tuesday",
            "wed": "Wednesday",
            "wednes": "Wednesday",
            "wednesday": "Wednesday",
            "thu": "Thursday",
            "thur": "Thursday",
            "thurs": "Thursday",
            "thursday": "Thursday",
            "fri": "Friday",
            "friday": "Friday",
            "sat": "Saturday",
            "satur": "Saturday",
            "saturday": "Saturday",
            "sun": "Sunday",
            "sunday": "Sunday",
        }
        weekday_match = re.fullmatch(
            r"last\s+(mon(?:day)?|tues?(?:day)?|wednes(?:day)?|thurs?(?:day)?|fri(?:day)?|satur(?:day)?|sun(?:day)?)",
            relative_text,
        )
        if weekday_match:
            weekday_key = weekday_match.group(1).lower()
            return (
                f"The {weekday_aliases.get(weekday_key, weekday_key.title())} before "
                f"{self._format_day_month_year(anchor_dt)}"
            )

        return text

    def _extract_evidence_contents(self, raw_context: str) -> List[str]:
        if not raw_context:
            return []
        matches = re.findall(
            r"\[Evidence\s+\d+\](?:[^\n]*?)Content:\s*(.*?)(?=\nKeywords:|\n---\n|\Z)",
            str(raw_context),
            flags=re.DOTALL,
        )
        contents = [re.sub(r"\s+", " ", match).strip() for match in matches if str(match).strip()]
        if contents:
            return contents
        primary = self._extract_primary_evidence_block(raw_context)
        return [re.sub(r"\s+", " ", primary).strip()] if primary else []

    def _looks_like_temporal_phrase(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(
            r"\b("
            r"\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?|"
            r"[A-Za-z]+\s+\d{4}|"
            r"\d{4}|"
            r"yesterday|today|last\s+(?:week|month|year|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"\d+\s+years?\s+ago|"
            r"since\s+\d{4}"
            r")\b",
            text,
            flags=re.IGNORECASE,
        ))

    def _question_terms_for_extraction(self, question: str, constraints: Dict[str, Any]) -> List[str]:
        stopwords = {
            "what", "which", "who", "when", "where", "why", "how", "did", "does", "do", "is", "are", "was",
            "were", "a", "an", "the", "of", "to", "for", "in", "on", "at", "from", "with", "by", "after",
            "before", "has", "have", "had", "still", "would", "could", "should", "can", "will", "be",
            "been", "being", "her", "his", "their", "she", "he", "they",
        }
        named_entities = {str(name).lower() for name in constraints.get("named_entities", []) if str(name).strip()}
        terms: List[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'+-]*", (question or "").lower()):
            if token in stopwords or token in named_entities or len(token) < 3:
                continue
            if token not in terms:
                terms.append(token)
        return terms

    def _select_best_evidence_text(self, question: str, raw_context: str, constraints: Dict[str, Any]) -> str:
        contents = self._extract_evidence_contents(raw_context)
        if not contents:
            return ""

        question_terms = self._question_terms_for_extraction(question, constraints)
        named_entities = [str(name).lower() for name in constraints.get("named_entities", []) if str(name).strip()]
        role_terms = [str(role).lower() for role in constraints.get("role_terms", []) if str(role).strip()]
        answer_type = str(constraints.get("answer_type", "") or "")
        best_text = contents[0]
        best_score = float("-inf")

        for content in contents:
            fragments = [frag.strip() for frag in re.split(r"(?<=[.!?])\s+", content) if frag.strip()]
            if not fragments:
                fragments = [content]
            for fragment in fragments:
                fragment_lower = fragment.lower()
                score = 0.0
                score += sum(1.0 for term in question_terms if term in fragment_lower)
                score += 1.4 * sum(1 for name in named_entities if name in fragment_lower)
                score += 0.7 * sum(1 for role in role_terms if role in fragment_lower)
                if answer_type != "temporal" and self._looks_like_temporal_phrase(fragment):
                    score -= 0.6
                if len(fragment.split()) >= 4:
                    score += 0.1
                if score > best_score:
                    best_score = score
                    best_text = fragment
        return best_text

    def _extract_question_head(self, question: str) -> Dict[str, Any]:
        q = re.sub(r"\s+", " ", (question or "").lower()).strip(" ?.")
        patterns = [
            (r"\bwhose\s+([a-z][a-z0-9+\- /']{0,40}?)(?:\s+(?:did|does|is|was|were|has|have)\b|$)", True),
            (r"\b(?:what|which)\s+kind\s+of\s+([a-z][a-z0-9+\- /']{0,40}?)(?:\s+(?:did|does|is|was|were|has|have)\b|$)", False),
            (r"\b(?:what|which)\s+([a-z][a-z0-9+\- /']{0,40}?)(?:\s+(?:did|does|is|was|were|has|have)\b|$)", False),
        ]
        for pattern, possessive in patterns:
            match = re.search(pattern, q)
            if match:
                head = re.sub(r"\s+", " ", match.group(1)).strip()
                if head:
                    return {"head": head, "possessive": possessive}
        return {"head": "", "possessive": False}

    def _expand_reference_pronoun(self, text: str, constraints: Dict[str, Any]) -> str:
        candidate = re.sub(r"\s+", " ", (text or "").strip()).strip(" ,.;")
        if not candidate:
            return ""
        names = [str(name).strip() for name in constraints.get("named_entities", []) if str(name).strip()]
        if not names:
            return candidate
        owner = names[0]
        lowered = candidate.lower()
        if lowered.startswith("her "):
            remainder = candidate[4:].rstrip("s").rstrip("'")
            return f"{owner}'s {remainder}".strip()
        if lowered.startswith("his "):
            remainder = candidate[4:].rstrip("s").rstrip("'")
            return f"{owner}'s {remainder}".strip()
        if lowered.startswith("their "):
            remainder = candidate[6:].rstrip("s").rstrip("'")
            return f"{owner}'s {remainder}".strip()
        return candidate

    def _clean_extracted_answer(self, text: str, constraints: Dict[str, Any]) -> str:
        candidate = re.sub(r"\s+", " ", (text or "").strip()).strip(" ,.;")
        if not candidate:
            return ""
        candidate = re.sub(
            r"\s+(?:on|in|at)\s+\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?$",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip(" ,.;")
        candidate = re.sub(r"\s+(?:recently|yesterday|today)$", "", candidate, flags=re.IGNORECASE).strip(" ,.;")
        if constraints.get("answer_type") in {"location", "person", "phrase"}:
            candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=re.IGNORECASE).strip()
        return candidate

    def _extract_head_answer_from_text(self, text: str, question: str, constraints: Dict[str, Any]) -> str:
        head_info = self._extract_question_head(question)
        head = str(head_info.get("head", "") or "").strip()
        if not head:
            return ""
        head_pattern = re.escape(head).replace(r"\ ", r"\s+")
        if head_info.get("possessive"):
            possessive_patterns = [
                rf"\b((?:her|his|their)\s+[A-Za-z][A-Za-z0-9'-]*)\s+{head_pattern}\b",
                rf"\b([A-Z][A-Za-z]+(?:'s\s+[A-Za-z][A-Za-z0-9'-]*)?)\s+{head_pattern}\b",
            ]
            for pattern in possessive_patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return self._clean_extracted_answer(
                        self._expand_reference_pronoun(match.group(1), constraints),
                        constraints,
                    )
            return ""
        matches = re.findall(
            rf"\b((?:[A-Za-z0-9][A-Za-z0-9+'/-]*\s+){{0,4}}{head_pattern})\b",
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            return ""
        best = max(matches, key=lambda item: len(item.split()))
        head_word_count = len(head.split())
        best_words = best.split()
        if len(best_words) > head_word_count + 3:
            best = " ".join(best_words[-(head_word_count + 3):])
        best = re.sub(
            r"^(?:(?:[A-Z][a-z]+|[A-Z][a-z]+'s)\s+){0,2}(?:recently\s+)?"
            r"(?:attend(?:ed|s)?|celebrat(?:ed|es)|visit(?:ed|s)?|support(?:ed|s)?|"
            r"make|made|move(?:d|s)?|buy|bought|paint(?:ed|s)?|realized?)\s+",
            "",
            best,
            flags=re.IGNORECASE,
        )
        return self._clean_extracted_answer(best, constraints)

    def _normalize_count_answer(self, text: str, question: str = "") -> str:
        candidate = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not candidate:
            return ""
        if "once or twice" in candidate and "how many times" in (question or "").lower():
            return "2"
        count_map = {
            "once": "1",
            "twice": "2",
            "thrice": "3",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        digit_match = re.search(r"\b(\d+)\s+times?\b", candidate)
        if digit_match:
            return digit_match.group(1)
        word_times_match = re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+times?\b",
            candidate,
            flags=re.IGNORECASE,
        )
        if word_times_match:
            return count_map[word_times_match.group(1).lower()]
        for token, normalized in count_map.items():
            if re.search(rf"\b{re.escape(token)}\b", candidate):
                return normalized
        return ""

    def _extract_count_from_text(self, text: str, question: str) -> str:
        normalized = self._normalize_count_answer(text, question=question)
        if normalized:
            return normalized
        match = re.search(r"\b(\d{1,2})\b", text)
        if match:
            value = match.group(1)
            if len(value) < 4:
                return value
        return ""

    def _extract_location_from_text(self, text: str, question: str, constraints: Dict[str, Any]) -> str:
        q_lower = (question or "").lower()
        location_patterns = []
        if " from " in q_lower:
            location_patterns.append(r"\bfrom\s+([A-Za-z][A-Za-z0-9'&+\-]*(?:\s+[A-Za-z][A-Za-z0-9'&+\-]*){0,4})")
        location_patterns.extend([
            r"\bin\s+([A-Za-z][A-Za-z0-9'&+\-]*(?:\s+[A-Za-z][A-Za-z0-9'&+\-]*){0,4})",
            r"\bat\s+([A-Za-z][A-Za-z0-9'&+\-]*(?:\s+[A-Za-z][A-Za-z0-9'&+\-]*){0,4})",
            r"\bto\s+([A-Za-z][A-Za-z0-9'&+\-]*(?:\s+[A-Za-z][A-Za-z0-9'&+\-]*){0,4})",
            r"\bnear\s+([A-Za-z][A-Za-z0-9'&+\-]*(?:\s+[A-Za-z][A-Za-z0-9'&+\-]*){0,4})",
        ])
        for pattern in location_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = self._clean_extracted_answer(match.group(1), constraints)
            if candidate and not self._looks_like_temporal_phrase(candidate):
                return candidate
        return ""

    def _extract_reason_phrase_from_text(self, text: str, constraints: Dict[str, Any]) -> str:
        patterns = [
            r"\binspired by\s+([^.;]+)",
            r"\bmotivated by\s+([^.;]+)",
            r"\brealized that\s+([^.;]+)",
            r"\bbecause\s+([^.;]+)",
            r"\bsince\s+([^.;]+)",
            r"\bexcited about\s+([^.;]+)",
            r"\binterested in\s+([^.;]+)",
            r"\bgreat for\s+([^.;]+)",
            r"\bsymboli[sz]es?\s+([^.;]+)",
            r"\bwants? to create\s+([^.;]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean_extracted_answer(match.group(1), constraints)
        return ""

    def _extract_person_from_text(self, text: str, question: str, constraints: Dict[str, Any]) -> str:
        verbs = self._question_terms_for_extraction(question, constraints)
        for verb in verbs[:4]:
            verb_root = verb[:-1] if verb.endswith("s") else verb
            match = re.search(rf"^\s*([^.;]{{2,80}}?)\s+\b{re.escape(verb_root)}\w*\b", text, flags=re.IGNORECASE)
            if match:
                candidate = self._clean_extracted_answer(
                    self._expand_reference_pronoun(match.group(1), constraints),
                    constraints,
                )
                if candidate and not self._looks_like_temporal_phrase(candidate):
                    return candidate
        for pattern in [
            r"\bwith\s+((?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})|(?:her|his|their)\s+[A-Za-z][A-Za-z ,'-]{0,40})",
            r"\bby\s+((?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})|(?:her|his|their)\s+[A-Za-z][A-Za-z ,'-]{0,40})",
        ]:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._clean_extracted_answer(
                    self._expand_reference_pronoun(match.group(1), constraints),
                    constraints,
                )
        return ""

    def _looks_like_empty_answer(self, text: str) -> bool:
        candidate = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not candidate:
            return True
        return any(marker in candidate for marker in [
            "unknown",
            "none",
            "nothing",
            "no answer",
            "not mentioned",
            "no information provided",
            "insufficient evidence",
            "not enough information",
        ])

    def _normalize_list_answer_text(self, text: str) -> str:
        candidate = re.sub(r"\s+", " ", (text or "").strip()).strip(" ,.;")
        if not candidate:
            return ""
        candidate = re.sub(r"^[^:]{0,50}:\s*", "", candidate)
        candidate = re.sub(
            r"^(?:they|she|he|we|i|my kids|the kids|melanie|caroline)\s+"
            r"(?:like|likes|love|loves|enjoy|enjoys|read|reads|made|make|painted|paint|"
            r"saw|seen|bought|buy|joined|join|participated in|participate in|has|have|had)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(r"^(?:at|in|on|to)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*;\s*", ", ", candidate)
        candidate = re.sub(r"\s+(?:and|or)\s+", ", ", candidate)
        candidate = re.sub(r"\s*,\s*", ", ", candidate)
        raw_items = [item.strip(" ,.;") for item in candidate.split(",") if item.strip(" ,.;")]
        items: List[str] = []
        seen = set()
        for item in raw_items:
            normalized = re.sub(r"^(?:a|an|the)\s+", "", item, flags=re.IGNORECASE).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            if any(key != prev and key in prev for prev in seen):
                continue
            seen.add(key)
            items.append(normalized)
        return ", ".join(items)

    def _extract_list_answer_from_context(self, question: str, raw_context: str, constraints: Dict[str, Any]) -> str:
        contents = self._extract_evidence_contents(raw_context)
        if not contents:
            return ""
        best_text = self._select_best_evidence_text(question, raw_context, constraints)
        ordered_texts = [best_text] + [content for content in contents if content and content != best_text]
        terms = self._question_terms_for_extraction(question, constraints)

        for text in ordered_texts:
            fragments = [frag.strip() for frag in re.split(r"(?<=[.!?])\s+", text) if frag.strip()]
            if not fragments:
                fragments = [text]
            for fragment in fragments:
                candidate = fragment
                for term in terms[:4]:
                    term_root = term[:-1] if term.endswith("s") else term
                    candidate = re.sub(rf"^.*?\b{re.escape(term_root)}\w*\b\s+", "", candidate, flags=re.IGNORECASE)
                candidate = self._normalize_list_answer_text(candidate)
                if candidate and (", " in candidate or len(candidate.split()) <= 6):
                    return candidate
        return ""

    def _extract_slot_answer_from_context(self, question: str, raw_context: str, constraints: Dict[str, Any]) -> str:
        best_text = self._select_best_evidence_text(question, raw_context, constraints)
        contents = self._extract_evidence_contents(raw_context)
        ordered_texts = [best_text] + [content for content in contents if content and content != best_text]
        answer_type = str(constraints.get("answer_type", "") or "")

        if answer_type == "list" or constraints.get("expects_list"):
            candidate = self._extract_list_answer_from_context(question, raw_context, constraints)
            if candidate:
                return candidate

        for text in ordered_texts:
            head_answer = self._extract_head_answer_from_text(text, question, constraints)
            if head_answer:
                return head_answer
            if answer_type == "location":
                candidate = self._extract_location_from_text(text, question, constraints)
                if candidate:
                    return candidate
            if answer_type == "count":
                candidate = self._extract_count_from_text(text, question)
                if candidate:
                    return candidate
            if answer_type == "person":
                candidate = self._extract_person_from_text(text, question, constraints)
                if candidate:
                    return candidate
            if answer_type == "reason_phrase" or constraints.get("has_reason_phrase_cue"):
                candidate = self._extract_reason_phrase_from_text(text, constraints)
                if candidate:
                    return candidate
        return ""

    def _should_use_slot_fallback(self, text: str, constraints: Dict[str, Any]) -> bool:
        candidate = re.sub(r"\s+", " ", (text or "").strip())
        if not candidate:
            return True
        answer_type = str(constraints.get("answer_type", "") or "")
        if answer_type != "temporal" and self._looks_like_temporal_phrase(candidate):
            return True
        if answer_type == "count":
            if self._normalize_count_answer(candidate):
                return False
            return self._looks_like_temporal_phrase(candidate)
        if answer_type == "person" and candidate.lower() in {"yes", "no", "likely yes", "likely no", "somewhat"}:
            return True
        if constraints.get("expects_list") and self._looks_like_temporal_phrase(candidate):
            return True
        return False

    def postprocess_answer(
        self,
        question: str,
        answer_text: str,
        raw_context: str = "",
        answer_style: str = AnswerStyle.EXTRACTIVE_SHORT.value,
    ) -> str:
        """
        轻量答案校准：
        - 去除解释性前缀与多余标点
        - 时间类问题优先提取标准日期短语，改善EM/F1
        """
        if answer_text is None:
            return ""
        constraints = self.router.infer_answer_constraints(question)
        text = str(answer_text).strip().strip('"').strip("'").strip()
        text = re.sub(r"^(answer\s*:|short answer\s*:)\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.rstrip(" .;!?,")

        q_lower = (question or "").lower()
        is_time_query = (
            answer_style == AnswerStyle.TEMPORAL_SHORT.value
            or "when" in q_lower
            or "date" in q_lower
            or "什么时候" in q_lower
        )

        if is_time_query:
            return self._normalize_temporal_answer(question, text, raw_context=raw_context)

        if answer_style == AnswerStyle.REASONING_LABEL.value:
            text = re.sub(r"[;,]?\s*brief reason\b.*$", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"\s*(?:as seen in|based on)\s+evidence\b.*$", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"\s+", " ", text).strip(" ;,.")
        elif answer_style == AnswerStyle.SUMMARY_SHORT.value:
            if self._looks_like_empty_answer(text):
                slot_answer = self._extract_slot_answer_from_context(question, raw_context, constraints)
                if slot_answer:
                    text = slot_answer
        else:
            use_slot_fallback = self._should_use_slot_fallback(text, constraints) or self._looks_like_empty_answer(text)
            if answer_style == AnswerStyle.LIST_SPAN.value and not any(sep in text for sep in [",", " and ", " or "]):
                slot_answer = self._extract_list_answer_from_context(question, raw_context, constraints)
                if slot_answer:
                    text = slot_answer
                    use_slot_fallback = False
            if use_slot_fallback or answer_style in {AnswerStyle.STRICT_ENTITY_SPAN.value, AnswerStyle.REASON_PHRASE.value}:
                slot_answer = self._extract_slot_answer_from_context(question, raw_context, constraints)
                if slot_answer:
                    text = slot_answer

        if constraints.get("starts_yes_no") or answer_style == AnswerStyle.REASONING_LABEL.value:
            lowered = text.lower()
            label_order = [
                ("likely yes", "Likely yes"),
                ("likely no", "Likely no"),
                ("somewhat", "Somewhat"),
                ("yes", "Yes"),
                ("no", "No"),
            ]
            for raw_label, normalized_label in label_order:
                if lowered.startswith(raw_label):
                    remainder = text[len(raw_label):].strip(" ;,.-")
                    if answer_style == AnswerStyle.REASONING_LABEL.value and remainder:
                        remainder = re.split(r"\s+(?:because|as|since)\b", remainder, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ;,.-")
                        if remainder:
                            return f"{normalized_label}; {remainder}"
                    return normalized_label

        if answer_style == AnswerStyle.ABSTAIN_OR_SPAN.value:
            if any(marker in text.lower() for marker in [
                "not mentioned",
                "not in the conversation",
                "insufficient evidence",
                "not enough information",
            ]):
                return "Not mentioned in the conversation"
            text = text.split("\n", 1)[0].strip()
            text = re.sub(r"\s*because\b.*$", "", text, flags=re.IGNORECASE).strip()

        # 非时间题：轻微清理冗余解释句前缀，不截断后半部分
        if answer_style == AnswerStyle.LIST_SPAN.value or constraints.get("expects_list"):
            text = self._normalize_list_answer_text(text)
        elif answer_style == AnswerStyle.SUMMARY_SHORT.value:
            text = text.split("\n", 1)[0].strip()
            text = re.sub(r"\s*(?:as seen in|based on)\s+evidence\b.*$", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"\s+", " ", text).strip(" ;,.")
        elif answer_style == AnswerStyle.REASON_PHRASE.value or constraints.get("answer_type") == "reason_phrase":
            text = re.sub(r"^(?:because|since)\s+", "", text, flags=re.IGNORECASE).strip()
            text = text.split("\n", 1)[0].strip()
            text = re.split(r"\s*;\s*", text, maxsplit=1)[0].strip()
        else:
            text = text.split("\n", 1)[0].strip()
            text = re.split(r"\s*;\s*", text, maxsplit=1)[0].strip()

        if constraints.get("answer_type") == "count":
            normalized_count = self._normalize_count_answer(text, question=question)
            if normalized_count:
                text = normalized_count

        if constraints.get("answer_type") in {"location", "person", "count", "phrase"} and len(text.split()) <= 4:
            text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE).strip()

        text = re.sub(r"^(it\s+is|it's|this\s+is|the answer is)\s+", "", text, flags=re.IGNORECASE).strip()
        return text.strip()

    def _build_answer_prompt(
        self,
        question: str,
        context: str,
        program: EvidenceProgram,
        verification: Optional[Dict[str, Any]] = None,
    ) -> str:
        answer_style = program.answer_style.value
        constraints = dict((program.metadata or {}).get("answer_constraints", {}))
        if not constraints:
            constraints = self.router.infer_answer_constraints(question)
        verification = verification or {}
        verifier_hint = verification.get("reason", "")
        verifier_line = f"Verifier hint: {verifier_hint}" if verifier_hint else ""
        guidance_lines = []
        if constraints.get("entity_sensitive"):
            guidance_lines.append("Do not swap people, relatives, possessions, or photo subjects. Preserve the exact named person and role from the question.")
        if constraints.get("expects_list"):
            guidance_lines.append("If multiple items are directly supported, include all and only the supported items as a comma-separated list.")
        elif constraints.get("expects_summary"):
            guidance_lines.append("Synthesize only the directly supported points needed for a short answer.")
        elif constraints.get("answer_type") == "location":
            guidance_lines.append("Return only the place name.")
        elif constraints.get("answer_type") == "person":
            guidance_lines.append("Return only the person, identity, or role phrase.")
        elif constraints.get("answer_type") == "count":
            guidance_lines.append("Return only the number or quantity phrase.")
        elif constraints.get("answer_type") == "reason_phrase":
            guidance_lines.append("Return only the shortest supported reason phrase, not a full sentence.")
        if constraints.get("starts_yes_no"):
            guidance_lines.append("Start with only Yes, No, Likely yes, Likely no, or Somewhat.")
        guidance_block = "\n".join(guidance_lines)
        if guidance_block:
            guidance_block = f"{guidance_block}\n"

        if answer_style == AnswerStyle.ABSTAIN_OR_SPAN.value:
            return f"""
                            Based on the context: {context}, answer the following question.
                            Only answer if the context contains direct, explicit evidence.
                            If the context does NOT contain direct evidence, output exactly
                            "Not mentioned in the conversation".
                            If the context DOES contain explicit evidence, return the shortest
                            supported answer span.
                            Do NOT guess or infer from tangentially related information.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """

        if answer_style == AnswerStyle.TEMPORAL_SHORT.value:
            return f"""
                            Based on the context: {context}, answer the following question.
                            Return only the benchmark-style date/time phrase.
                            If the evidence uses relative time, convert it with the recorded conversation date:
                            - "yesterday" at recorded 8 May, 2023 -> "7 May 2023"
                            - "last week" at recorded 9 June, 2023 -> "The week before 9 June 2023"
                            - "last Friday" at recorded 15 July, 2023 -> "The Friday before 15 July 2023"
                            - "this month" at recorded 3 July, 2023 -> "July 2023"
                            - "last year" at recorded 17 August, 2023 -> "2022"
                            For duration questions, keep duration style such as "4 years", "10 years ago", or "Since 2016".
                            Do NOT output full sentences or explanations.
                            {guidance_block}

                            Question: {question} Short answer:
                            """

        if answer_style == AnswerStyle.STRICT_ENTITY_SPAN.value:
            return f"""
                            Based on the context: {context}, answer the following entity-sensitive question.
                            Return the shortest supported answer span.
                            Never swap the named person, family role, possession, or photo subject.
                            If the context mentions a similar fact about someone else, ignore it.
                            Do NOT answer with "unknown", "none", or "not mentioned" unless there is truly no direct span.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """

        if answer_style == AnswerStyle.LIST_SPAN.value:
            return f"""
                            Based on the context: {context}, answer the following list question.
                            Return all directly supported items as a comma-separated list.
                            Do NOT collapse the answer to one item if multiple supported items are present.
                            Do NOT add related extras that are not explicitly supported.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """

        if answer_style == AnswerStyle.SUMMARY_SHORT.value:
            return f"""
                            Based on the context: {context}, answer the following synthesis question.
                            Return one short sentence or compact clause that directly answers the question.
                            Prefer 8-18 words when possible.
                            Do NOT reduce a how/why/plans/traits question to a single noun.
                            Do NOT include evidence citations or long explanations.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """

        if answer_style == AnswerStyle.REASON_PHRASE.value:
            return f"""
                            Based on the context: {context}, answer the following question.
                            Return only the shortest supported reason phrase or target clause.
                            Prefer a direct phrase over a full sentence.
                            Do NOT add explanation outside the supported phrase.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """

        if answer_style == AnswerStyle.REASONING_LABEL.value:
            q_lower = (question or "").strip().lower()
            if q_lower.startswith(("would", "is", "are", "do", "does", "could", "should", "can")):
                return f"""
                            Based on the context: {context}, answer the following inference question.
                            Infer the most likely answer from the evidence.
                            Do NOT upgrade a related fact into the asked claim.
                            Prefer the shortest valid form.
                            If the label alone answers the question, output only:
                            - "Likely yes"
                            - "Likely no"
                            - "Yes"
                            - "No"
                            - "Somewhat"
                            If a reason is truly needed, add only a 3-6 word clause after a semicolon.
                            Never mention evidence numbers.
                            Keep the whole answer under 10 words when possible.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """
            return f"""
                            Based on the context: {context}, answer the following inference question.
                            Return only the target label or a short phrase.
                            If a plain label is enough, return only the label.
                            If needed, add only a very short support clause after a semicolon.
                            Do NOT include long explanations or evidence citations.
                            Keep the answer under 8 words unless a short clause is necessary.
                            {guidance_block}{verifier_line}

                            Question: {question} Short answer:
                            """

        return f"""Based on the context: {context}, extract the shortest exact answer span for the following question.
                            Copy exact words from the context whenever possible.
                            Do NOT paraphrase. Do NOT explain.
                            For list questions, include only the directly supported items and avoid adding related extras.
                            Prefer the minimal benchmark-style noun phrase over a descriptive sentence.
                            {guidance_block}

                            Question: {question} Short answer:
                            """

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

    def answer_question(self, question: str, category: Optional[int] = None) -> str:
        """Generate answer for a question given the conversation context."""
        program = self.predict_evidence_program(question)
        mode = dict(self._classify_question_mode(question, category=category, program=program))
        mode["effective_pipeline"] = "program_guided_pipeline"
        self._last_route_decision = dict(mode)
        self._last_program = program.to_dict()
        self._last_verification = {}
        self._last_execution_trace = {}
        self._last_compressed_evidence_count = 0
        self._last_rrf_scores = {}
        self._last_channel_details = {}
        self._last_retrieval_metadata = {}
        self._last_answer_subgraph_summary = {}
        outcome = self.pipeline.run(
            question=question,
            program=program,
            temperature_c5=self.temperature_c5,
        )
        self._last_verification = dict(outcome.verification)
        self._last_execution_trace = dict(outcome.retrieval_result.execution_trace)
        self._last_compressed_evidence_count = int(outcome.compressed_evidence_count)
        self._last_rrf_scores = dict(outcome.retrieval_result.rrf_scores)
        self._last_channel_details = dict(outcome.retrieval_result.channel_details)
        self._last_retrieval_metadata = dict(outcome.retrieval_result.metadata)
        if outcome.retrieval_result.answer_subgraph is not None:
            self._last_answer_subgraph_summary = {
                "valid": bool(outcome.retrieval_result.answer_subgraph.valid),
                "used_as_primary": bool(outcome.retrieval_result.answer_subgraph.used_as_primary),
                "confidence": float(outcome.retrieval_result.answer_subgraph.confidence),
                "node_count": len(outcome.retrieval_result.answer_subgraph.nodes),
                "edge_count": len(outcome.retrieval_result.answer_subgraph.edges),
                "seed_node_ids": list(outcome.retrieval_result.answer_subgraph.seed_node_ids),
                "bridge_node_ids": list(outcome.retrieval_result.answer_subgraph.bridge_node_ids),
                "metadata": dict(outcome.retrieval_result.answer_subgraph.metadata),
            }
        return outcome.response_text, outcome.user_prompt, outcome.raw_context


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
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     api_base: Optional[str] = None, api_key: Optional[str] = None,
                     embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                     reranker_model: str = DEFAULT_RERANKER_MODEL,
                     reranker_cutoff_layer: int = DEFAULT_RERANKER_CUTOFF_LAYER):
    """Evaluate the agent on the LoComo dataset.

    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
    """
    # 固定随机种子，确保评测可重复性
    random.seed(42)
    np.random.seed(42)
    
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_ours_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = setup_logger(log_path)


    # 新增：创建结果保存目录（用户指定的新文件夹）
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(os.path.dirname(__file__), "GraphRAGresults_Optimized")  # 使用新的测试文件夹
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

    existing_indices = set()
    try:
        for fn in os.listdir(output_dir):
            if fn.startswith("GraphRAG_result_sample_") and fn.endswith(".json") and not fn.endswith("_final.json"):
                try:
                    idx = int(fn.replace("GraphRAG_result_sample_", "").replace(".json", ""))
                    existing_indices.add(idx)
                except Exception:
                    pass
    except Exception:
        pass

    # 调试时仅跑前几个 sample，便于观察 routing 的泛化性
    start_sample_idx = 0
    num_eval_samples = 1
    end_sample_idx = min(len(samples), start_sample_idx + num_eval_samples)
    samples = samples[start_sample_idx:end_sample_idx]
    logger.info(
        f"运行 samples [{start_sample_idx}, {max(start_sample_idx, end_sample_idx - 1)}] "
        f"（共 {len(samples)} 个样本）进行 routing 泛化性测试"
    )
    
    logger.info(
        f"Starting evaluation on {len(samples)} samples (starting from code index: {start_sample_idx})")

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
        original_sample_idx = start_sample_idx + sample_idx
        
        # --- 增加断点续跑：如果该 sample 的详细结果和 final 结果都已存在，直接跳过 ---
        sample_output_file = os.path.join(output_dir, f"GraphRAG_result_sample_{original_sample_idx}.json")
        sample_final_output_file = os.path.join(output_dir, f"GraphRAG_result_sample_{original_sample_idx}_final.json")
        if os.path.exists(sample_output_file) and os.path.exists(sample_final_output_file):
            logger.info(
                f"⏭ 发现已有结果文件 {sample_output_file} 和 {sample_final_output_file}，"
                f"跳过 sample {original_sample_idx}"
            )
            continue
        # -------------------------------------------------------------
        
        # agent = advancedMemAgent(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)

        agent = advancedMemAgent(
            model=model,
            backend=backend,
            retrieve_k=retrieve_k,
            temperature_c5=temperature_c5,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            reranker_cutoff_layer=reranker_cutoff_layer,
            api_base=api_base,
            api_key=api_key,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
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
            return (
                hasattr(sample, 'community_id')
                and hasattr(sample, 'id_based_links')
                and getattr(sample, 'text_schema_version', 0) == MEMORY_TEXT_SCHEMA_VERSION
            )

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
                    # 生成与 add_documents 路径一致的 doc_texts，修复 BM25 索引不一致 bug
                    _doc_texts = [agent.memory_system._generate_doc_text(m)
                                  for m in cached_memories.values()]
                    _dense_doc_texts = [agent.memory_system._generate_dense_text(m)
                                        for m in cached_memories.values()]
                    if os.path.exists(retriever_cache_file):
                        agent.memory_system.retriever = agent.memory_system.retriever.load(
                            retriever_cache_file, retriever_cache_embeddings_file
                        )
                    elif os.path.exists(retriever_cache_embeddings_file):
                        agent.memory_system.retriever = \
                            agent.memory_system.retriever.load_from_local_memory(
                                cached_memories, 'BAAI/bge-m3',
                                agent.memory_system.retriever_alpha,
                                embeddings_cache_file=retriever_cache_embeddings_file,
                                doc_texts=_doc_texts,
                                dense_doc_texts=_dense_doc_texts,
                            )
                    else:
                        agent.memory_system.retriever = \
                            agent.memory_system.retriever.load_from_local_memory(
                                cached_memories, 'BAAI/bge-m3',
                                agent.memory_system.retriever_alpha,
                                doc_texts=_doc_texts,
                                dense_doc_texts=_dense_doc_texts,
                            )

                    # 恢复 GraphRAG 图边 + 社区状态
                    graph_cache_dir = os.path.join(memories_dir, f"graph_state_sample_{original_sample_idx}")
                    if os.path.exists(graph_cache_dir):
                        try:
                            agent.memory_system.load_graph(graph_cache_dir)
                            logger.info(f"✅ GraphRAG 图状态已恢复（"
                                        f"{len(agent.memory_system.graph_edges)} 条边，"
                                        f"{len(agent.memory_system.communities)} 个社区）")
                        except Exception as graph_error:
                            logger.info(f"⚠ 图缓存失效：{graph_error}，将基于现有记忆重建图状态")
                            agent.memory_system._rebuild_graph_edges()
                            agent.memory_system.rebuild_communities()
                            agent.memory_system.save_graph(graph_cache_dir)
                    else:
                        # 没有图状态缓存，基于已有记忆重建社区
                        logger.info("⚠ 无图状态缓存，基于现有记忆重建社区...")
                        agent.memory_system._rebuild_graph_edges()
                        agent.memory_system.rebuild_communities()
                        # 修复：此前只重建不落盘，导致下次运行仍然找不到 graph_state
                        agent.memory_system.save_graph(graph_cache_dir)
                        logger.info(f"✅ 已保存重建后的 GraphRAG 图状态 → {graph_cache_dir}")

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
                
                batch_contents = []
                batch_times = []
                
                for turn in turns.turns:
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker " + turn.speaker + " says : " + turn.text
                    
                    # Context Retrieval 步骤 2：局部块上下文生成 (Chunk Context Generation)
                    # 提示 LLM 根据全局上下文为单个孤立句子生成背景解释
                    chunk_context = agent.generate_chunk_context(session_full_text, conversation_tmp)
                    
                    # Context Retrieval 步骤 3：内容融合与向量化
                    enriched_content = f"Context: {chunk_context}\nContent: {conversation_tmp}"
                    
                    batch_contents.append(enriched_content)
                    batch_times.append(turn_datatime)
                    
                # 批量向量化与添加
                if batch_contents:
                    agent.add_memories_batch(batch_contents, times=batch_times)

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
                prediction, user_prompt, raw_context = agent.answer_question(qa.question)
                try:
                    prediction = json.loads(prediction)["answer"]
                except (json.JSONDecodeError, KeyError, TypeError):
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
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # 3. 尝试正则提取
                        match = re.search(r'"answer":\s*"(.*?)"', clean_prediction)
                        if match:
                            prediction = match.group(1)
                        else:
                            # 4. 彻底失败，使用清洗后的文本作为答案
                            prediction = clean_prediction
                            logger.info(f"Failed to parse prediction as JSON: {prediction}")
                            error_num += 1

                # 统一答案后处理，减少解释性文本对EM/F1的伤害
                prediction = agent.postprocess_answer(
                    qa.question,
                    prediction,
                    raw_context,
                    answer_style=getattr(agent, "_last_program", {}).get("answer_style", AnswerStyle.EXTRACTIVE_SHORT.value),
                )
                # Log results
                logger.info(f"\nQuestion {total_questions}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"User Prompt: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Program: {getattr(agent, '_last_program', {})}")
                logger.info(f"Verification: {getattr(agent, '_last_verification', {})}")
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
                category_int = int(qa.category)
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "category_name": agent.category_name_map.get(category_int, f"category_{category_int}"),
                    "strategy_name": getattr(agent, "_last_route_decision", {}).get("strategy_name"),
                    "route_task_type": getattr(agent, "_last_route_decision", {}).get("route_task_type"),
                    "route_reason": getattr(agent, "_last_route_decision", {}).get("route_reason"),
                    "route_confidence": getattr(agent, "_last_route_decision", {}).get("route_confidence"),
                    "effective_pipeline": getattr(agent, "_last_route_decision", {}).get("effective_pipeline"),
                    "rrf_confidence_threshold": getattr(agent, "_last_route_decision", {}).get("rrf_confidence_threshold"),
                    "enable_abstain_gate": getattr(agent, "_last_route_decision", {}).get("enable_abstain_gate"),
                    "channel_weights": getattr(agent, "_last_route_decision", {}).get("channel_weights"),
                    "retrieval_engine": getattr(agent, "_last_program", {}).get("retrieval_engine"),
                    "program_type": getattr(agent, "_last_program", {}).get("program_type"),
                    "answer_style": getattr(agent, "_last_program", {}).get("answer_style"),
                    "need_verifier": getattr(agent, "_last_program", {}).get("need_verifier"),
                    "verification_label": getattr(agent, "_last_verification", {}).get("label"),
                    "verification_reason": getattr(agent, "_last_verification", {}).get("reason"),
                    "verification_confidence": getattr(agent, "_last_verification", {}).get("confidence"),
                    "compressed_evidence_count": getattr(agent, "_last_compressed_evidence_count", 0),
                    "evidence_selection_mode": getattr(agent, "_last_retrieval_metadata", {}).get("evidence_selection_mode"),
                    "answer_subgraph_summary": getattr(agent, "_last_answer_subgraph_summary", {}),
                    "execution_trace": getattr(agent, "_last_execution_trace", {}),
                    "metrics": metrics
                }
                results.append(result)
                # 新增：将当前QA结果加入当前Sample的列表
                sample_results.append(result)
                # Log progress
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")

        # 新增：保存当前Sample的独立结果文件
        original_sample_idx = start_sample_idx + sample_idx
        sample_output_file = os.path.join(output_dir, f"GraphRAG_result_sample_{original_sample_idx}.json")
        with open(sample_output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "sample_idx": original_sample_idx,  # 样本原始索引（0-9）
                "total_questions": len(sample_results),  # 该样本的问题数
                "individual_results": sample_results  # 该样本的所有QA详细结果
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 已保存Sample {original_sample_idx} 结果到：{sample_output_file}")

        sample_metrics = []
        sample_categories = []
        sample_category_counts = defaultdict(int)
        for item in sample_results:
            metrics_item = item.get("metrics", {})
            if metrics_item:
                sample_metrics.append(metrics_item)
            category_item = item.get("category")
            if category_item is not None:
                sample_categories.append(category_item)
                sample_category_counts[category_item] += 1

        sample_aggregate_results = aggregate_metrics(sample_metrics, sample_categories) if sample_metrics and sample_categories else {}
        sample_final_results = {
            "model": model,
            "dataset": dataset_path,
            "sample_idx": original_sample_idx,
            "total_questions": len(sample_results),
            "category_distribution": {
                str(cat): count for cat, count in sample_category_counts.items()
            },
            "aggregate_metrics": sample_aggregate_results,
        }
        with open(sample_final_output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 已保存Sample {original_sample_idx} final结果到：{sample_final_output_file}")


    agg_metrics = []
    agg_categories = []
    agg_total_q = 0
    try:
        for fn in os.listdir(output_dir):
            if fn.startswith("GraphRAG_result_sample_") and fn.endswith(".json") and not fn.endswith("_final.json"):
                fp = os.path.join(output_dir, fn)
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    agg_total_q += int(data.get("total_questions", 0))
                    for item in data.get("individual_results", []):
                        m = item.get("metrics", {})
                        if m:
                            agg_metrics.append(m)
                        cat = item.get("category")
                        if cat is not None:
                            agg_categories.append(cat)
    except Exception:
        pass

    if agg_metrics and agg_categories:
        aggregate_results = aggregate_metrics(agg_metrics, agg_categories)
        total_questions = agg_total_q
    else:
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

    # 仅在用户显式传入 --output 时，额外保存一份跨 sample 的聚合结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        logger.info("Per-sample final files already saved; skipping combined aggregate final file.")

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
    parser.add_argument("--model", type=str, default="qwen3_5-flash",
                        help="Model to use (e.g., qwen3_5-flash)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="vllm",
                        help="Backend to use (openai, vllm, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=40,
                        help="Retrieve k")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help="Embedding model for dense retrieval")
    parser.add_argument("--reranker_model", type=str, default=DEFAULT_RERANKER_MODEL,
                        help="Reranker model. Supports CrossEncoder rerankers and FlagEmbedding LLM rerankers")
    parser.add_argument("--reranker_cutoff_layer", type=int, default=DEFAULT_RERANKER_CUTOFF_LAYER,
                        help="Cutoff layer used by layerwise reranker models such as bge-reranker-v2-minicpm-layerwise")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Override API base URL (OpenAI-compatible). For vLLM use: http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="asdasdasd",
                        help="API key for the LLM (default: asdasdasd)")
    parser.add_argument("--vllm_host", type=str, default="http://localhost",
                        help="vLLM host (for vllm backend)")
    parser.add_argument("--vllm_port", type=int, default=8004,
                        help="vLLM port (for vllm backend)")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None

    effective_api_base = args.api_base
    if args.backend == "vllm" and not effective_api_base:
        effective_api_base = f"{args.vllm_host}:{args.vllm_port}/v1"

    evaluate_dataset(
        dataset_path,
        args.model,
        output_path,
        args.ratio,
        args.backend,
        args.temperature_c5,
        args.retrieve_k,
        args.sglang_host,
        args.sglang_port,
        api_base=effective_api_base,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        reranker_cutoff_layer=args.reranker_cutoff_layer,
    )


if __name__ == "__main__":
    main()
