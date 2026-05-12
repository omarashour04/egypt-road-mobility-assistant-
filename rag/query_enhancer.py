"""
rag/query_enhancer.py
=====================
Query enhancement before FAISS retrieval. Two strategies:

1. HyDE (Hypothetical Document Embeddings)
   - Ask the LLM to write a hypothetical answer to the question
   - Embed THAT answer instead of the raw query
   - Rationale: the hypothetical answer uses domain vocabulary that matches
     real document chunks far better than a short user question does.
   - Especially effective for Arabic where the user might write colloquially
     ("كم غرامة الموبايل") but the law says ("استخدام الهاتف المحمول أثناء القيادة")

2. Bilingual query expansion
   - For Arabic queries: also search with an English translation
   - For English queries: also search with an Arabic translation
   - Merge FAISS results, deduplicate by chunk_id
   - Rationale: our corpus is bilingual; an Arabic question should find
     English chunks that answer it, and vice versa.

Both are opt-in via config flags. HyDE is the higher-value one.
"""

from __future__ import annotations

import ollama
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_BASE_URL,
    GENERATOR_MODEL_NAME,
    HYDE_ENABLED,
    HYDE_FALLBACK_ON_ERROR,
)


# ── Prompt templates ──────────────────────────────────────────────────────────

HYDE_PROMPT_AR = """/no_think
أنت خبير في قانون المرور المصري وقوانين الطرق والمركبات.

اكتب فقرة قصيرة (3-5 جمل) تجيب على السؤال التالي كما لو كانت مقتطفة من
وثيقة قانونية أو مقال رسمي مصري. استخدم مصطلحات قانونية رسمية.
لا تقل "لا أعرف" — اكتب إجابة افتراضية مقنعة.

السؤال: {question}

الإجابة الافتراضية:"""

HYDE_PROMPT_EN = """/no_think
You are an expert on Egyptian traffic law, vehicle regulations, and road rules.

Write a short paragraph (3-5 sentences) that answers the following question
as if it were extracted from an Egyptian legal document or official article.
Use formal legal terminology. Do NOT say "I don't know" — write a plausible answer.

Question: {question}

Hypothetical answer:"""

TRANSLATE_TO_EN_PROMPT = """/no_think
Translate the following Arabic question about Egyptian traffic law into English.
Return ONLY the translation, nothing else.

Arabic: {question}
English:"""

TRANSLATE_TO_AR_PROMPT = """/no_think
Translate the following English question about Egyptian traffic law into Arabic.
Return ONLY the translation, nothing else.

English: {question}
Arabic:"""


class QueryEnhancer:
    """
    Enhances queries before retrieval using HyDE and bilingual expansion.

    Main entry point: enhance(query) → returns one or more query strings
    to use for FAISS search. Caller should merge results from each.
    """

    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model  = GENERATOR_MODEL_NAME
        logger.info("QueryEnhancer initialized (HyDE + bilingual expansion)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_content(self, response) -> str:
        """Extract content from ollama.chat() — handles both dict and object responses."""
        if hasattr(response, "message"):
            msg = response.message
            return msg.content if hasattr(msg, "content") else msg.get("content", "")
        elif isinstance(response, dict):
            msg = response.get("message", {})
            return msg.get("content", "") if isinstance(msg, dict) else ""
        return ""

    @staticmethod
    def _is_arabic(text: str) -> bool:
        """
        Heuristic: if more than 30% of alpha characters are Arabic Unicode,
        treat the query as Arabic.
        """
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        total_chars  = sum(1 for c in text if c.isalpha())
        if total_chars == 0:
            return False
        return (arabic_chars / total_chars) > 0.3

    # ── Public API ────────────────────────────────────────────────────────────

    def get_hyde_query(self, question: str) -> str | None:
        """
        Generate a hypothetical document for HyDE retrieval.
        Returns the hypothetical text, or None if generation fails and
        HYDE_FALLBACK_ON_ERROR is True.
        """
        if not HYDE_ENABLED:
            return None

        is_arabic = self._is_arabic(question)
        prompt    = HYDE_PROMPT_AR if is_arabic else HYDE_PROMPT_EN
        prompt    = prompt.format(question=question)

        try:
            response   = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 200},
            )
            hypothesis = self._get_content(response).strip()

            if len(hypothesis) < 30:
                logger.warning(f"HyDE generated suspiciously short response: '{hypothesis}'")
                return None

            logger.debug(f"HyDE hypothesis ({len(hypothesis)} chars): {hypothesis[:80]}...")
            return hypothesis

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            if HYDE_FALLBACK_ON_ERROR:
                return None
            raise

    def get_translation(self, question: str) -> str | None:
        """
        Translate the question to the other language for bilingual retrieval.
        Arabic → English, English → Arabic.
        Returns translated string or None on failure.
        """
        is_arabic = self._is_arabic(question)
        prompt    = TRANSLATE_TO_EN_PROMPT if is_arabic else TRANSLATE_TO_AR_PROMPT
        prompt    = prompt.format(question=question)

        try:
            response    = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 100},
            )
            translation = self._get_content(response).strip()

            if translation.strip() == question.strip():
                return None

            logger.debug(f"Translation: '{question[:40]}' → '{translation[:40]}'")
            return translation

        except Exception as e:
            logger.warning(f"Query translation failed: {e}")
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
_enhancer_instance: QueryEnhancer | None = None


def get_query_enhancer() -> QueryEnhancer:
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = QueryEnhancer()
    return _enhancer_instance