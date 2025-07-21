from uagents import Agent
from chat_proto import chat_proto

agent = Agent(
    name="hart_image_gen_agent",
    seed="hartagentsecretphrase",
    mailbox=True,
    port=8001
)

agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
