from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

class StoryPrompt(BaseModel):
    prompt: str

@app.post("/generate_story")
def generate_story_api(story_prompt: StoryPrompt):
    """
    Generates a story using the Mixtral-8x7B-Instruct-v0.1 model.
    """
    generator = pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    story = generator(story_prompt.prompt, max_length=500, num_return_sequences=1)
    return {"story": story[0]["generated_text"]}

class UserInteraction(BaseModel):
    user_id: str
    story_prompt: str
    rating: str

@app.post("/learn_preference")
def learn_preference_api(interaction: UserInteraction):
    """
    This is a placeholder function to demonstrate how user preferences can be learned.
    In a real application, this would store the interaction in a database
    and use it to build a recommendation model.
    """
    print(f"Learning from user {interaction.user_id}'s interaction with story: '{interaction.story_prompt}'. Rating: {interaction.rating}")
    return {"status": "success"}
