# TODO: Implement Hybrid Translation Model for Multilingual Story Generation

The goal is to support story generation from prompts in multiple languages (e.g., Turkish, English, Chinese).

## Plan

1.  **Add Translation Library:**
    *   Add the `googletrans-py` library to the `requirements.txt` file.
    *   Install the new requirement (`pip install -r requirements.txt`).

2.  **Update `main.py` with Translation Logic:**
    *   Import the `Translator` from the `googletrans` library.
    *   In the `generate_story_api` function, implement the following logic:
        1.  **Detect Language:** Detect the language of the incoming prompt (`story_prompt.prompt`).
        2.  **Translate to English (if necessary):** If the detected language is not English, translate the prompt text to English.
        3.  **Generate Story:** Send the (now guaranteed to be English) prompt to the story generation model.
        4.  **Translate Back to Original Language (if necessary):** If the original prompt was not in English, translate the generated story back to the original language.
        5.  **Return Story:** Return the final story in the user's original language.

3.  **Testing:**
    *   Restart the `uvicorn` server.
    *   Test the endpoint with prompts in different languages (e.g., Turkish, English, and another language if possible) to ensure the translation and generation process works correctly.
