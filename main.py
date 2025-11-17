from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import Literal
import torch
import re

# Define a list of keywords that would make a story inappropriate for children
INAPPROPRIATE_KEYWORDS = [
    "death", "killed", "murder", "violence", "bloody", "sex", "sexual", "drugs", "alcohol",
    "curse", "swore", "swearing", "hate", "revenge", "torture", "scream", "terror", "fear",
    "monster", "demon", "devil", "ghost", "witch", "evil", "darkness", "nightmare",
    "weapon", "knife", "sword", "gun", "fight", "battle", "war", "destroy", "destroyer",
    "suffer", "pain", "agony", "cry", "weep", "sadness", "despair", "hopeless",
    "slave", "slavery", "prison", "jail", "crime", "criminal", "steal", "rob",
    "lie", "lying", "cheat", "cheating", "betray", "betrayal", "deceive", "deception",
    "greed", "jealousy", "envy", "pride", "wrath", "gluttony", "lust", "sloth",
    "hell", "damn", "f***" # Added common explicit words
]

def is_appropriate_content(story_text, keywords=INAPPROPRIATE_KEYWORDS):
    """
    Checks if the story text contains any inappropriate keywords.
    Returns True if appropriate, False otherwise.
    """
    story_text_lower = story_text.lower()
    for keyword in keywords:
        if keyword in story_text_lower:
            return False
    return True

app = FastAPI()

# Model and tokenizer names
base_model_name = "meta-llama/Meta-Llama-3.1-8B"
fine_tuned_model_path = "./llama_fine_tuned_model"

# BitsAndBytesConfig for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load the PEFT adapter
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model = model.merge_and_unload() # Merge LoRA layers into the base model for inference
model.eval() # Set model to evaluation mode

# Set tokenizer padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def post_process_story(text):
    # Remove the initial prompt injected by the model
    text = re.sub(r'^Write a (children[\'’]s|story for adults) story about:.*?\\n\\n', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^A (children[\'’]s|story for adults) story about:.*?\\n\\n', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove common phrases that are not part of the story (more aggressive)
    text = re.sub(r'Perfect for young fans of.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Parents should be advised that.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'This book is ideal for.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'A must-read for.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Winner of the.*?Award.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Graudin weaves an exciting plot.*?', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+ has managed to take the style of.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+ has such a way with words.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'Characters are spunky, smart-mouthed.*?', '', text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'---', '', text) # Remove the separator
    
    # Remove text in parentheses or brackets (could be noisy)
    text = re.sub(r'[\[\(\]][^\)\]]*[\)\]]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences and remove consecutive duplicates (simple repetition check)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    for sentence in sentences:
        if not cleaned_sentences or sentence.lower() != cleaned_sentences[-1].lower():
            cleaned_sentences.append(sentence)
    text = ' '.join(cleaned_sentences)

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text).strip()
    
    return text

class StoryPrompt(BaseModel):
    prompt: str
    story_type: Literal["children", "adult"] = "children"

@app.post("/generate_story")
async def generate_story_api(story_prompt: StoryPrompt):
    """
    Generates a story based on the selected story type (children or adult) using the fine-tuned LLaMA model.
    """
    if story_prompt.story_type == "children":
        full_prompt = f"Write a children's story about: {story_prompt.prompt}"
    else: # adult
        full_prompt = f"Write a story for adults about: {story_prompt}"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the story
    cleaned_story = post_process_story(generated_text)

    # Final check for inappropriate content
    if not is_appropriate_content(cleaned_story) and story_prompt.story_type == "children":
        return {"story": "I'm sorry, but I cannot generate a children's story with potentially inappropriate content. Please try a different prompt."}

    return {"story": cleaned_story}