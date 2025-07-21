import os
from uagents import Model
from gradio_client import Client

client = Client("https://hart.hanlab.ai/")

class ImageRequest(Model):
    prompt: str

class ImageResponse(Model):
    image_url: str


result = client.predict(
		prompt="Hello!!",
		seed=42,
		guidance_scale=4.5,
		randomize_seed=True,
		api_name="/run"
)
def generate_image(prompt: str) -> str:
    try:
        response = client.predict(
            prompt=prompt,
            api_name="/run"
        )
    except Exception as e:
        return f"An error occurred: {e}"
    return response
