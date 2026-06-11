import asyncio
import traceback
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob, Options
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful AI assistant that answers phone calls. Keep your responses concise and friendly.",
        )

    async def on_enter(self) -> None:
        await self.session.say("Hello! I'm your real-time assistant that created by Prince. How can I help you today?")

    async def on_exit(self) -> None:
        await self.session.say("Goodbye! It was great talking with you!")

async def start_session(context: JobContext):
    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(voice="Leda", response_modalities=["AUDIO"])
    )
    pipeline = Pipeline(llm=model)
    session = AgentSession(agent=MyVoiceAgent(), pipeline=pipeline)
    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    return JobContext(room_options=RoomOptions())

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