"""
memory_store.py — RAG-based conversation memory using ChromaDB Cloud + Gemini Embeddings

Two layers:
  1. In-call memory  : full message history kept in ConversationMemory (in-process)
  2. Cross-call memory: caller summaries stored as vector embeddings in ChromaDB Cloud,
                        retrieved by phone number on every new call.

Install:
    pip install chromadb google-generativeai python-dotenv

.env keys required:
    GOOGLE_API_KEY
    CHROMA_API_KEY
    CHROMA_TENANT        (e.g. 19fad97e-e364-4c4c-b63d-123b92e9f46c)
    CHROMA_DATABASE      (e.g. calling-agent)
"""

import os
import time
import logging
import asyncio
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("memory-store")

# ─── Gemini setup ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ChromaDB Gemini embedding function
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name="models/text-embedding-004",   # latest free Gemini embedding model
)

# ─── ChromaDB Cloud client ────────────────────────────────────────────────────
chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

# Single collection for all caller memories
caller_collection = chroma_client.get_or_create_collection(
    name="caller_memory",
    embedding_function=gemini_ef,
    metadata={"hnsw:space": "cosine"},
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. IN-CALL MEMORY  (ephemeral, lives only for the duration of one call)
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Maintains an ordered, per-call transcript."""

    def __init__(self, caller_phone: str, max_turns: int = 50):
        self.caller_phone = caller_phone
        self.max_turns = max_turns
        self._history: list[dict] = []   # {"role": "user"|"assistant", "text": ..., "ts": ...}
        self.call_start = datetime.utcnow().isoformat()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_user(self, text: str) -> None:
        self._push("user", text)

    def add_assistant(self, text: str) -> None:
        self._push("assistant", text)

    def get_history(self) -> list[dict]:
        return list(self._history)

    def format_for_prompt(self) -> str:
        """Return recent turns as a compact string for injection into the system prompt."""
        lines = []
        for turn in self._history[-20:]:           # last 20 turns
            role = "Caller" if turn["role"] == "user" else "Agent"
            lines.append(f"{role}: {turn['text']}")
        return "\n".join(lines)

    def summary_text(self) -> str:
        """Plain-text dump used for creating the post-call RAG document."""
        lines = [f"[Call on {self.call_start}]"]
        for t in self._history:
            role = "Caller" if t["role"] == "user" else "Agent"
            lines.append(f"  {role}: {t['text']}")
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────────────

    def _push(self, role: str, text: str) -> None:
        self._history.append({"role": role, "text": text, "ts": time.time()})
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-self.max_turns * 2:]   # trim oldest


# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-CALL RAG MEMORY  (persisted in ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class CallerMemoryStore:
    """
    Store and retrieve caller-specific memories via semantic search.

    Each document stored in ChromaDB represents one past call summary
    (or a key fact extracted from that call).
    """

    MAX_DOCS_PER_CALLER = 20   # keep the last N call summaries per caller

    # ── Store ─────────────────────────────────────────────────────────────────

    async def save_call(
        self,
        caller_phone: str,
        conversation: ConversationMemory,
    ) -> None:
        """
        Summarise the call with Gemini and upsert into ChromaDB.
        Call this at the end of every session.
        """
        raw_transcript = conversation.summary_text()
        if not raw_transcript.strip():
            return

        summary = await self._summarise(raw_transcript)
        doc_id = f"{caller_phone}_{int(time.time())}"

        caller_collection.add(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{
                "caller_phone": caller_phone,
                "call_start": conversation.call_start,
                "raw_length": len(raw_transcript),
            }],
        )
        logger.info("[memory] saved call summary for %s (doc_id=%s)", caller_phone, doc_id)

        # Prune old docs to avoid unbounded growth
        await self._prune(caller_phone)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    async def recall(
        self,
        caller_phone: str,
        query: str = "previous conversations",
        top_k: int = 5,
    ) -> str:
        """
        Return a formatted string of the most relevant past-call summaries
        for injection into the system prompt at the start of a new call.
        """
        try:
            results = caller_collection.query(
                query_texts=[query],
                n_results=min(top_k, self._count(caller_phone)),
                where={"caller_phone": caller_phone},
            )
        except Exception as exc:
            logger.warning("[memory] recall failed: %s", exc)
            return ""

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return ""

        lines = ["=== Previous interactions with this caller ==="]
        for doc, meta in zip(docs, metas):
            lines.append(f"[{meta.get('call_start', 'unknown date')}]\n{doc}")
        lines.append("=== End of history ===")
        return "\n\n".join(lines)

    async def get_caller_profile(self, caller_phone: str) -> str:
        """
        High-level caller profile: retrieve ALL summaries and ask Gemini
        to distill them into a short profile paragraph.
        """
        try:
            results = caller_collection.get(
                where={"caller_phone": caller_phone},
            )
        except Exception:
            return ""

        docs = results.get("documents", [])
        if not docs:
            return ""

        combined = "\n\n".join(docs)
        prompt = (
            "Below are summaries of past phone calls with the same caller. "
            "Write a concise 3-sentence profile about this person: their name (if known), "
            "common topics, preferences, and any important context the agent should know.\n\n"
            + combined
        )
        profile = await self._gemini_text(prompt)
        return profile

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _summarise(self, transcript: str) -> str:
        prompt = (
            "Summarise the following phone call transcript in 3-5 sentences. "
            "Include: caller's name (if mentioned), key topics discussed, any decisions or "
            "requests made, and important facts for future calls.\n\n"
            + transcript
        )
        return await self._gemini_text(prompt)

    async def _prune(self, caller_phone: str) -> None:
        """Delete oldest docs if we exceed MAX_DOCS_PER_CALLER."""
        try:
            results = caller_collection.get(
                where={"caller_phone": caller_phone},
                include=["metadatas"],
            )
        except Exception:
            return

        ids = results.get("ids", [])
        metas = results.get("metadatas", [])
        if len(ids) <= self.MAX_DOCS_PER_CALLER:
            return

        # Sort by call_start ascending, delete oldest
        paired = sorted(zip(ids, metas), key=lambda x: x[1].get("call_start", ""))
        to_delete = [pid for pid, _ in paired[:len(ids) - self.MAX_DOCS_PER_CALLER]]
        caller_collection.delete(ids=to_delete)
        logger.info("[memory] pruned %d old docs for %s", len(to_delete), caller_phone)

    def _count(self, caller_phone: str) -> int:
        try:
            res = caller_collection.get(where={"caller_phone": caller_phone}, include=[])
            return len(res.get("ids", []))
        except Exception:
            return 0

    @staticmethod
    async def _gemini_text(prompt: str) -> str:
        loop = asyncio.get_event_loop()
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = await loop.run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        return response.text.strip()


# ── Singleton ──────────────────────────────────────────────────────────────────
caller_memory_store = CallerMemoryStore()