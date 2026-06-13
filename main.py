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
    if not phone:
        return ""
    try:
        res = memory_collection.get(where={"phone": phone})
        docs = res.get("documents", [])
        return "\n".join(docs) if docs else ""
    except Exception as e:
        logger.error(f"Error fetching memory: {e}")
        return ""


def search_knowledge(query: str):
    if not query:
        return ""
    try:
        res = knowledge_collection.query(
            query_texts=[query],
            n_results=3
        )
        docs = res.get("documents", [[]])
        return "\n".join(docs[0]) if docs and docs[0] else ""
    except Exception as e:
        logger.error(f"Error in knowledge search: {e}")
        return ""


def save_memory(phone: str, text: str):
    if not phone:
        return
    try:
        memory_collection.add(
            ids=[str(uuid4())],
            documents=[text],
            metadatas=[{"phone": phone, "created_at": datetime.utcnow().isoformat()}]
        )
        logger.info(f"Memory saved for phone: {phone}")
    except Exception as e:
        logger.error(f"Error saving memory: {e}")


def save_summary(phone: str, text: str):
    try:
        summary_collection.add(
            ids=[str(uuid4())],
            documents=[text],
            metadatas=[{"phone": phone, "created_at": datetime.utcnow().isoformat()}]
        )
        logger.info(f"Summary saved for phone: {phone}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")


# =========================
# AGENT
# =========================

class MyVoiceAgent(Agent):
    def __init__(self, phone: str = "", memory: str = "", knowledge: str = ""):
        self.caller_phone = phone

        context_block = ""
        if memory:
            context_block += f"\nCUSTOMER MEMORY (from previous calls):\n{memory}\n"
        if knowledge:
            context_block += f"\nKNOWLEDGE BASE:\n{knowledge}\n"

        super().__init__(
            instructions=f"""
You are Sakshi, Medisaver AI Healthcare Assistant.
{context_block}
Rules:
- Speak short, natural Hindi/English/Hinglish.
- Ask one question at a time.
- Always use "Ji" after customer name.
- Never give medical advice.
- Be polite, calm, human-like.
- Use customer memory if available to personalize the conversation.
- Use knowledge base facts if relevant to the query.
- Do not hallucinate or make up information.
"""
        )

    async def on_enter(self):
        await self.session.say(
            "Hello, I'm Sakshi from Medisaver. May I know your name?"
        )

    async def on_exit(self):
        await self.session.say(
            "Thank you Ji. Our team will contact you if needed. Goodbye!"
        )


# =========================
# SESSION
# =========================

async def start_session(context: JobContext):

    # -------------------------------------------------------
    # STEP 1: Try to extract phone from room metadata
    # JobContext does NOT support .on() event hooks —
    # phone is captured later via STT or room participant list
    # -------------------------------------------------------
    phone = ""
    try:
        room_opts = context.room_options
        if room_opts and hasattr(room_opts, "metadata") and room_opts.metadata:
            phone = room_opts.metadata.get("phone", "")
            logger.info(f"Phone from room metadata: {phone}")
    except Exception as e:
        logger.warning(f"Could not extract phone from context: {e}")

    logger.info(f"Caller phone at session start: {phone or 'unknown'}")

    # -------------------------------------------------------
    # STEP 2: Load RAG context upfront using phone (if known)
    # -------------------------------------------------------
    memory = get_memory(phone)
    knowledge = search_knowledge("medisaver services plans pricing")
    logger.info(f"Memory loaded: {bool(memory)} | Knowledge loaded: {bool(knowledge)}")

    # -------------------------------------------------------
    # STEP 3: Build agent with injected context
    # -------------------------------------------------------
    agent = MyVoiceAgent(phone=phone, memory=memory, knowledge=knowledge)

    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        # api_key read automatically from GOOGLE_API_KEY in .env
        config=GeminiLiveConfig(
            voice="Kore",
            response_modalities=["AUDIO"]
        )
    )

    pipeline = Pipeline(llm=model)
    session = AgentSession(agent=agent, pipeline=pipeline)

    # =========================
    # STATE
    # =========================
    state = {
        "phone": phone,
        "latest_transcript": "",
        "transcripts": []
    }

    # =========================
    # STT HOOK
    # =========================
    @pipeline.on("stt")
    async def on_transcript(text: str):
        logger.info(f"[STT] {text}")

        state["latest_transcript"] = text
        state["transcripts"].append(text)

        # Try to capture phone from SIP participant ID via room participants
        if not state["phone"]:
            try:
                participants = session.room.participants if hasattr(session, "room") and session.room else {}
                for pid, p in participants.items():
                    pid_str = str(pid)
                    if pid_str.startswith("+") or (pid_str.lstrip("+").isdigit() and len(pid_str) > 7):
                        state["phone"] = pid_str
                        logger.info(f"Phone captured from room participants: {pid_str}")
                        break
            except Exception as e:
                logger.debug(f"Could not get phone from participants: {e}")

        # Save memory when user introduces themselves
        name_triggers = [
            "my name", "i am", "call me", "naam hai",
            "mera naam", "naam mera", "main hoon", "mai hoon", "myself"
        ]
        if any(k in text.lower() for k in name_triggers):
            if state["phone"]:
                save_memory(state["phone"], text)
            else:
                logger.info(f"Name info captured but phone unknown: {text}")

        return text

    # =========================
    # LLM HOOK — log only (S2S: no messages[] to inject into)
    # =========================
    @pipeline.on("llm")
    async def on_llm(prompt: dict):
        logger.info(f"[LLM] {repr(prompt)}")
        return prompt

    # =========================
    # TTS HOOK
    # =========================
    @pipeline.on("tts")
    async def on_tts(text: str):
        logger.info(f"[TTS] {text}")
        return text

    # =========================
    # TURN HOOKS
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
    # POST-CALL: Save summary
    # =========================
    try:
        full_conversation = "\n".join(state["transcripts"])
        if full_conversation:
            save_summary(
                phone=state["phone"] or "unknown",
                text=full_conversation
            )
            logger.info("Call summary saved.")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")


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