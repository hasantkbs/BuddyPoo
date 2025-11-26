# BuddyPoo

BuddyPoo is an AI-powered storytelling tool for children. It generates creative and engaging stories to spark imagination and make reading fun.

## Current Features

*   **AI-powered story generation:** Creates unique stories based on user prompts using a Large Language Model (currently configured for Llama-3.1 8B Instruct, but can be switched).
*   **FastAPI web service:** Exposes story generation as a REST API endpoint.

## Getting Started

To get started with BuddyPoo, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [repository_url]
    cd BuddyPoo
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the FastAPI application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    (Note: This will run in the foreground. For background, add `&` at the end and manage the process accordingly.)

5.  **Test the API (example using curl):**
    ```bash
    curl -X POST "http://0.0.0.0:8000/generate_story" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Once upon a time, there was a brave knight named Sir Reginald who embarked on a quest to save a princess from a dragon. What happened next?"}'
    ```

## Contributing

(Coming soon)