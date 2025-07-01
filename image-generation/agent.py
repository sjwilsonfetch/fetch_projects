import os
from enum import Enum

from uagents import Agent, Context, Model
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage

from chat_proto import chat_proto
from models import ImageRequest, ImageResponse, generate_image

AGENT_SEED = os.getenv("AGENT_SEED", "image-generator-agent-seed-phrase")
AGENT_NAME = os.getenv("AGENT_NAME", "Image Generator Agent")

PORT = 8000
agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=PORT,
    mailbox=True,
)


# Include protocol
agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
