"""
rag/generator.py
================
LLM answer generation via Ollama (qwen3:4b).
Environment: ollama binary 0.23.2, Python client 0.6.2

qwen3:4b thinking chain structure:
  [English reasoning] → [draft Arabic answer] → [English self-correction]
  → [revised Arabic answer] → [more English] → [FINAL Arabic answer]

We always take the LAST Arabic paragraph — it is always the final, best answer.
Taking multiple paragraphs leaks the reasoning process into the output.
"""

import logging
import re

import ollama

from config import (
    GENERATOR_MAX_NEW_TOKENS,
    GENERATOR_MODEL_NAME,
    GENERATOR_TEMPERATURE,
    GENERATOR_THINKING_MODE,
    OLLAMA_BASE_URL,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_WITH_HISTORY,
)

logger = logging.getLogger(__name__)


def _is_arabic(text: str) -> bool:
    if not text:
        return False
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
    return arabic / len(text) > 0.20


class Generator:
    def __init__(self) -> None:
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.model  = GENERATOR_MODEL_NAME
        logger.info(f"Generator initialised | model={self.model}")

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        history_block: str = "",
        thinking: bool | None = None,
    ) -> dict:
        use_thinking = thinking if thinking is not None else GENERATOR_THINKING_MODE
        arabic = _is_arabic(question)

        # ── Build context ─────────────────────────────────────────────────────
        if not context_chunks:
            context = "No relevant information found in the knowledge base."
        else:
            parts = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get("source_name", "unknown")
                group  = chunk.get("group", "")
                text   = chunk.get("text", "")
                parts.append(f"[{i}] Source: {source} ({group})\n{text}")
            context = "\n\n".join(parts)

        # ── Build prompt ──────────────────────────────────────────────────────
        if history_block:
            prompt = PROMPT_TEMPLATE_WITH_HISTORY.format(
                history=history_block,
                context=context,
                question=question,
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )

        # Ensure correct think directive at top
    
        # Force output language
        if arabic:
            prompt += "\n\nمهم: أجب باللغة العربية فقط."
        else:
            prompt += "\n\nIMPORTANT: Answer in English only."

        # ── Call Ollama ───────────────────────────────────────────────────────
        logger.debug(f"Generating | arabic={arabic} | thinking={use_thinking} | chunks={len(context_chunks)}")

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": GENERATOR_TEMPERATURE,
                    "num_predict": max(GENERATOR_MAX_NEW_TOKENS, 800),
                },
            )

            content, thinking_text = self._parse_response(response)
            logger.debug(f"content={len(content)}c | thinking={len(thinking_text)}c")
            answer = self._extract_answer(content, thinking_text, arabic)

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            answer = ""

        # ── Fallback ──────────────────────────────────────────────────────────
        if not answer.strip():
            logger.error("Empty answer — returning fallback")
            answer = (
                "لا تتوفر معلومات كافية في المصادر المتاحة."
                if arabic else
                "Insufficient information in the available sources."
            )

        return {
            "answer":        answer,
            "model":         self.model,
            "thinking_used": use_thinking,
        }

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response) -> tuple[str, str]:
        content  = ""
        thinking = ""

        if hasattr(response, "message"):
            msg      = response.message
            content  = (getattr(msg, "content",  None) or "").strip()
            thinking = (getattr(msg, "thinking", None) or "").strip()

        if not content and not thinking:
            try:
                d        = response.model_dump() if hasattr(response, "model_dump") else {}
                msg_d    = d.get("message", {}) or {}
                content  = (msg_d.get("content",  "") or "").strip()
                thinking = (msg_d.get("thinking", "") or "").strip()
            except Exception:
                pass

        if not content and not thinking and isinstance(response, dict):
            msg_d    = response.get("message", {}) or {}
            content  = (msg_d.get("content",  "") or "").strip()
            thinking = (msg_d.get("thinking", "") or "").strip()

        return content, thinking

    # ── Answer extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(content: str, thinking: str, arabic: bool) -> str:
        # Content field populated — use directly
        if content:
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return cleaned if cleaned else content.strip()

        # Extract from thinking field
        if thinking:
            return Generator._extract_from_thinking(thinking, arabic)

        return ""

    @staticmethod
    def _extract_from_thinking(thinking: str, arabic: bool) -> str:
        """
        Return ONLY the last paragraph in the target language.

        qwen3 revises its answer multiple times during thinking. The very last
        Arabic paragraph is always the final, polished answer. Taking more than
        one paragraph risks including mid-reasoning drafts.
        """
        if not thinking:
            return ""

        paragraphs = [p.strip() for p in thinking.split("\n\n") if p.strip()]

        # Walk backwards — return the first paragraph we find in target language
        # that is substantive (>= 8 words) and not a reasoning phrase
        for para in reversed(paragraphs):
            if len(para.split()) < 8:
                continue

            if arabic:
                if _is_arabic(para):
                    return para
            else:
                if not _is_arabic(para):
                    lower = para.lower()
                    is_reasoning = any(lower.startswith(p) for p in [
                        "let me", "i need", "i think", "i'll", "actually",
                        "but ", "wait,", "hmm,", "looking at", "the question",
                        "based on my", "i should", "so i", "first,", "okay,",
                        "now,", "the user", "they want", "this is about",
                    ])
                    if not is_reasoning:
                        return para

        # Last resort: second half of thinking
        mid = len(thinking) // 2
        return thinking[mid:].strip()


# ── Singleton ─────────────────────────────────────────────────────────────────
_generator_instance: Generator | None = None


def get_generator() -> Generator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = Generator()
    return _generator_instance