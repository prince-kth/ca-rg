import asyncio
import traceback
import os
import logging
import chromadb
from uuid import uuid4
from datetime import datetime

from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob, Options
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medisaver-rag")


# =========================
# CHROMA CLOUD
# =========================

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant="19fad97e-e364-4c4c-b63d-123b92e9f46c",
    database="calling-agent"
)

knowledge_collection = chroma_client.get_or_create_collection("knowledge_base")
memory_collection = chroma_client.get_or_create_collection("customer_memory")
summary_collection = chroma_client.get_or_create_collection("call_summaries")


# =========================
# RAG HELPERS
# =========================

def get_memory(phone: str):
    try:
        res = memory_collection.get(where={"phone": phone})
        docs = res.get("documents", [])
        return "\n".join(docs) if docs else ""
    except:
        return ""


def search_knowledge(query: str):
    try:
        res = knowledge_collection.query(
            query_texts=[query],
            n_results=3
        )
        docs = res.get("documents", [[]])
        return "\n".join(docs[0]) if docs and docs[0] else ""
    except:
        return ""


def save_memory(phone: str, text: str):
    memory_collection.add(
        ids=[str(uuid4())],
        documents=[text],
        metadatas=[{"phone": phone, "created_at": datetime.utcnow().isoformat()}]
    )


def save_summary(phone: str, text: str):
    summary_collection.add(
        ids=[str(uuid4())],
        documents=[text],
        metadatas=[{"phone": phone, "created_at": datetime.utcnow().isoformat()}]
    )


# =========================
# AGENT
# =========================

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are Sakshi, Medisaver AI Healthcare Assistant.

Rules:
- Speak short, natural Hindi/English/Hinglish.
- Ask one question at a time.
- Always use "Ji" after customer name.
- Never give medical advice.
- Be polite, calm, human-like.
"""
        )

    async def on_enter(self):
        await self.session.say(
            "Hello, I'm Sakshi from Medisaver. May I know your name?"
        )

    async def on_exit(self):
        await self.session.say(
            "Thank you Ji. Our team will contact you if needed."
        )


# =========================
# SESSION
# =========================

async def start_session(context: JobContext):

    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(
            voice="Kore",
            response_modalities=["AUDIO"]
        )
    )

    pipeline = Pipeline(llm=model)
    session = AgentSession(agent=MyVoiceAgent(), pipeline=pipeline)


    # =========================
    # STATE (simple memory)
    # =========================
    state = {
        "phone": None,
        "latest_transcript": ""
    }


    # =========================
    # 🔊 STT HOOK
    # =========================
    @pipeline.on("stt")
    async def on_transcript(text: str):
        logger.info(f"[STT] {text}")

        state["latest_transcript"] = text

        # simple trigger: store memory when user gives info
        if any(k in text.lower() for k in ["my name", "i am", "call me"]):
            if state["phone"]:
                save_memory(state["phone"], text)

        return text


    # =========================
    # 🤖 BEFORE LLM (MAIN RAG POINT)
    # =========================
    @pipeline.on("llm")
    async def before_llm(prompt: dict):

        user_text = state.get("latest_transcript", "")

        phone = state.get("phone", "")

        memory = get_memory(phone)
        knowledge = search_knowledge(user_text)

        enriched_context = f"""
        CUSTOMER MEMORY:
        {memory}

        KNOWLEDGE BASE:
        {knowledge}
        """

        prompt["messages"].insert(0, {
            "role": "system",
            "content": f"""
You are Sakshi (Medisaver AI Assistant).

Use this context:
{enriched_context}

Rules:
- Use memory if available
- Use knowledge if relevant
- Do not hallucinate
"""
        })

        return prompt


    # =========================
    # 🔊 TTS HOOK
    # =========================
    @pipeline.on("tts")
    async def on_tts(text: str):
        logger.info(f"[TTS] {text}")
        return text


    # =========================
    # TURN MANAGEMENT
    # =========================
    @pipeline.on("user_turn_end")
    async def user_end():
        logger.info("User finished speaking")


    @pipeline.on("agent_turn_end")
    async def agent_end():
        logger.info("Agent response done")


    # =========================
    # START SESSION
    # =========================
    await session.start(
        wait_for_participant=True,
        run_until_shutdown=True
    )


# =========================
# CONTEXT
# =========================

def make_context():
    return JobContext(
        room_options=RoomOptions()
    )


# =========================
# RUN
# =========================

if __name__ == "__main__":
    try:
        options = Options(
            agent_id="MyTelephonyAgent",
            register=True,
            max_processes=10,
            host="localhost",
            port=8081,
        )

        job = WorkerJob(
            entrypoint=start_session,
            jobctx=make_context,
            options=options
        )

        job.start()

    except Exception:
        traceback.print_exc()