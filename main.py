from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from typing import Literal

app = FastAPI()

class StoryPrompt(BaseModel):
    prompt: str
    story_type: Literal["children", "adult"] = "children"

@app.post("/generate_story")
def generate_story_api(story_prompt: StoryPrompt):
    """
    Generates a story based on the selected story type (children or adult).
    """
    
    if story_prompt.story_type == "children":
        full_prompt = f"Write a children's story about: {story_prompt.prompt}"
    else: # adult
        full_prompt = f"Write a story for adults about: {story_prompt.prompt}"

    generator = pipeline("text-generation", model="./fine_tuned_model")
    story = generator(full_prompt, max_new_tokens=500, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.7, no_repeat_ngram_size=2)
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
