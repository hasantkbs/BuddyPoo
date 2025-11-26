from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

generator = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto")

class StoryPrompt(BaseModel):
    prompt: str

@app.post("/generate_story")
def generate_story_api(story_prompt: StoryPrompt) -> dict:
    """
    Generates a story using the BLOOM model.
    """
    story = generator(story_prompt.prompt, max_length=500, num_return_sequences=1)
    return {"story": story[0]["generated_text"]}