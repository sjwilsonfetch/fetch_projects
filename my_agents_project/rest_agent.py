import time
from typing import Any, Dict
from uagents import Agent, Context, Model

# Define your models
class Request(Model):
    text: str

class Response(Model):
    timestamp: int
    text: str
    agent_address: str

# Create your agent
agent = Agent(name="Rest API")

# GET endpoint example
@agent.on_rest_get("/rest/get", Response)
async def handle_get(ctx: Context) -> Dict[str, Any]:
    ctx.logger.info("Received GET request")
    return {
        "timestamp": int(time.time()),
        "text": "Hello from the GET handler!",
        "agent_address": ctx.agent.address,
    }

# POST endpoint example
@agent.on_rest_post("/rest/post", Request, Response)
async def handle_post(ctx: Context, req: Request) -> Response:
    ctx.logger.info("Received POST request")
    return Response(
        text=f"Received: {req.text}",
        agent_address=ctx.agent.address,
        timestamp=int(time.time()),
    )

if __name__ == "__main__":
    agent.run()
