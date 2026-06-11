import asyncio
import time
import traceback
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob, Options
from videosdk.agents.event_bus import global_event_emitter
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telephony-debug")
load_dotenv()


def _log_room_state(context: JobContext, label: str) -> None:
    room = context.room
    if not room:
        logger.warning("[%s] room not available yet", label)
        return

    participants = getattr(room, "participants_data", {}) or {}
    logger.info("[%s] room_id=%s participants=%s", label, getattr(room, "room_id", "?"), len(participants))
    for pid, info in participants.items():
        logger.info(
            "[%s] participant id=%s name=%s sipUser=%s sipCallType=%s",
            label,
            pid,
            info.get("name"),
            info.get("sipUser"),
            info.get("sipCallType"),
        )

    sip_manager = getattr(room, "sip_manager", None)
    session_id = getattr(room, "session_id", None)
    if sip_manager and session_id:
        try:
            call_info = sip_manager.fetch_call_info(session_id=session_id)
            logger.info("[%s] sip_call_info=%s", label, call_info)
        except Exception as exc:
            logger.warning("[%s] could not fetch sip call info: %s", label, exc)


def _install_debug_hooks(t0: float) -> None:
    def on_audio_stream_enabled(data):
        stream = data.get("stream")
        participant = data.get("participant")
        logger.info(
            "[event] AUDIO_STREAM_ENABLED +%.2fs | participant=%s kind=%s",
            time.perf_counter() - t0,
            getattr(participant, "display_name", None),
            getattr(stream, "kind", None),
        )

    def on_participant_left(data):
        participant = data.get("participant")
        logger.info(
            "[event] PARTICIPANT_LEFT +%.2fs | participant=%s",
            time.perf_counter() - t0,
            getattr(participant, "display_name", None),
        )

    def on_agent_started(data):
        logger.info("[event] AGENT_STARTED +%.2fs", time.perf_counter() - t0)

    def on_pipeline_error(data):
        logger.error("[event] PIPELINE_ERROR +%.2fs | %s", time.perf_counter() - t0, data)

    global_event_emitter.on("AUDIO_STREAM_ENABLED", on_audio_stream_enabled)
    global_event_emitter.on("PARTICIPANT_LEFT", on_participant_left)
    global_event_emitter.on("AGENT_STARTED", on_agent_started)
    global_event_emitter.on("PIPELINE_ERROR", on_pipeline_error)


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful AI assistant that answers phone calls. Keep your responses concise and friendly.",
        )

    async def on_enter(self) -> None:
        logger.info("[agent] on_enter called — sending greeting")
        await self.session.say("Hello! I'm your real-time assistant. How can I help you today?")
        logger.info("[agent] greeting sent")

    async def on_exit(self) -> None:
        logger.info("[agent] on_exit called")
        await self.session.say("Goodbye! It was great talking with you!")


async def start_session(context: JobContext):
    t0 = time.perf_counter()
    _install_debug_hooks(t0)
    logger.info("[session] job started")

    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"]),
    )
    pipeline = Pipeline(llm=model)
    session = AgentSession(agent=MyVoiceAgent(), pipeline=pipeline)

    shutdown_event = asyncio.Event()

    async def cleanup_session():
        logger.info("[session] cleanup started +%.2fs", time.perf_counter() - t0)
        try:
            await asyncio.wait_for(session.close(), timeout=15.0)
        except Exception as exc:
            logger.error("[session] error closing session: %s", exc)
        shutdown_event.set()

    context.add_shutdown_callback(cleanup_session)

    def on_session_end(reason: str):
        logger.info("[event] SESSION_END +%.2fs | reason=%s", time.perf_counter() - t0, reason)
        asyncio.create_task(context.shutdown())

    await context.connect()
    logger.info("[session] room connected +%.2fs", time.perf_counter() - t0)
    _log_room_state(context, "after-connect")

    if context.room:
        context.room.setup_session_end_callback(on_session_end)

    logger.info("[session] waiting for participant +%.2fs", time.perf_counter() - t0)
    participant_id = await context.wait_for_participant()
    if participant_id is None:
        logger.warning("[session] ended before participant joined")
        return

    logger.info("[session] participant ready id=%s +%.2fs", participant_id, time.perf_counter() - t0)
    _log_room_state(context, "after-participant")

    try:
        logger.info("[session] starting pipeline (Gemini init) +%.2fs", time.perf_counter() - t0)
        await session.start()
        logger.info("[session] agent session started +%.2fs", time.perf_counter() - t0)

        await shutdown_event.wait()
        logger.info("[session] shutdown complete +%.2fs", time.perf_counter() - t0)
    finally:
        try:
            await session.close()
        except Exception as exc:
            logger.error("[session] error in finally close: %s", exc)
        try:
            await context.shutdown()
        except Exception as exc:
            logger.error("[session] error in finally shutdown: %s", exc)


def make_context() -> JobContext:
    return JobContext(
        room_options=RoomOptions(
            playground=False,
            session_timeout_seconds=30,
        )
    )


if __name__ == "__main__":
    try:
        options = Options(
            agent_id="MyTelephonyAgent",  # CRITICAL: used for routing
            register=True,               # REQUIRED for telephony
            max_processes=10,
            host="localhost",
            port=8081,
        )
        job = WorkerJob(entrypoint=start_session, jobctx=make_context, options=options)
        job.start()
    except Exception as e:
        traceback.print_exc()
