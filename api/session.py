"""
api/session.py
==============
In-memory conversation session store with TTL-based expiry.

Each session holds the last SESSION_MAX_TURNS (Q, A) pairs, which are
prepended to the prompt so the model has conversational context.

Sessions are keyed by a UUID that the client receives on the first /ask
call and passes back in subsequent requests.

No external dependencies — pure Python with asyncio for cleanup.

Example flow:
    POST /ask {"question": "What is the fine for speeding?"}
    → Response: {"answer": "...", "session_id": "abc-123", ...}

    POST /ask {"question": "What about using a phone?", "session_id": "abc-123"}
    → Model sees previous Q&A in its prompt context
"""

from __future__ import annotations

import uuid
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SESSION_TTL_MINUTES, SESSION_MAX_TURNS


@dataclass
class Turn:
    """A single question-answer exchange."""
    question: str
    answer:   str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    """Conversation session — holds recent turns and tracks last activity."""
    session_id:  str
    turns:       list[Turn] = field(default_factory=list)
    created_at:  datetime   = field(default_factory=datetime.utcnow)
    last_active: datetime   = field(default_factory=datetime.utcnow)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        # Keep only the most recent SESSION_MAX_TURNS turns
        if len(self.turns) > SESSION_MAX_TURNS:
            self.turns = self.turns[-SESSION_MAX_TURNS:]
        self.last_active = datetime.utcnow()

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.last_active + timedelta(minutes=SESSION_TTL_MINUTES)

    def build_history_block(self) -> str:
        """
        Format previous turns as a conversation history block for the prompt.
        Returns empty string if no turns yet.
        """
        if not self.turns:
            return ""

        lines = ["Previous conversation:"]
        for turn in self.turns:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        lines.append("")   # blank line before current question

        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class SessionStore:
    """
    Thread-safe (asyncio) in-memory session store.

    Sessions expire after SESSION_TTL_MINUTES of inactivity.
    The cleanup task runs every 5 minutes to evict stale sessions.
    """

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._cleanup_task: asyncio.Task | None = None
        logger.info(
            f"SessionStore initialized "
            f"(TTL={SESSION_TTL_MINUTES}min, max_turns={SESSION_MAX_TURNS})"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_cleanup(self) -> None:
        """Start background cleanup task. Call from FastAPI lifespan."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Session cleanup task started")

    def stop_cleanup(self) -> None:
        """Cancel cleanup task. Call from FastAPI lifespan shutdown."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    # ── Session management ────────────────────────────────────────────────────

    def create_session(self) -> Session:
        """Create a new session and return it."""
        session_id = str(uuid.uuid4())
        session    = Session(session_id=session_id)
        self._sessions[session_id] = session
        logger.debug(f"New session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Session | None:
        """
        Retrieve a session by ID.
        Returns None if the session doesn't exist or has expired.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            logger.debug(f"Session expired: {session_id}")
            del self._sessions[session_id]
            return None
        return session

    def get_or_create(self, session_id: str | None) -> Session:
        """
        Get existing session or create a new one.
        This is the main entry point used by the /ask endpoint.
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
            # Session ID was provided but doesn't exist or expired
            logger.debug(f"Session not found or expired: {session_id}, creating new")

        return self.create_session()

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record a Q&A turn in a session."""
        session = self.get_session(session_id)
        if session:
            session.add_turn(question, answer)

    def get_history(self, session_id: str) -> list[dict]:
        """Return the turn history for a session as a list of dicts."""
        session = self.get_session(session_id)
        if not session:
            return []
        return [
            {
                "question":  t.question,
                "answer":    t.answer,
                "timestamp": t.timestamp.isoformat(),
            }
            for t in session.turns
        ]

    def delete_session(self, session_id: str) -> bool:
        """Explicitly delete a session. Returns True if it existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Session deleted: {session_id}")
            return True
        return False

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    # ── Background cleanup ────────────────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        """Evict expired sessions every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300)   # 5 minutes
                self._evict_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Session cleanup error: {e}")

    def _evict_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.debug(f"Evicted {len(expired)} expired session(s)")


# ── Singleton ─────────────────────────────────────────────────────────────────
_store_instance: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = SessionStore()
    return _store_instance
