# BuddyPoo - AI-Powered Story Generation

BuddyPoo is an AI-powered storytelling tool that leverages the LLaMA 3.1 8B model, fine-tuned with QLoRA/PEFT, to generate creative stories based on user prompts. The project aims to explore the capabilities of large language models in generating engaging and contextually relevant narratives.

## Features

*   **LLaMA 3.1 8B Core Model:** Utilizes the powerful LLaMA 3.1 8B model from Meta as the foundation for story generation.
*   **QLoRA/PEFT Fine-tuning:** The model is fine-tuned using QLoRA (Quantized Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-tuning) techniques on a curated dataset of children's literature to enhance its storytelling abilities.
*   **Customizable Story Prompts:** Users can provide prompts to guide the story generation process.
*   **API Endpoint:** Provides a FastAPI endpoint (`/generate_story`) for easy integration and access to the story generation functionality.
*   **Post-processing:** Includes a post-processing step to clean and refine the generated text, removing common model artifacts and improving readability.
*   **Age-Appropriate Content (Goal):** While the model is fine-tuned on children's literature, ensuring consistently age-appropriate and coherent content remains an ongoing area of improvement.

## Getting Started

Follow these steps to set up and run the BuddyPoo project locally.

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)
*   `kaggle` CLI (for downloading Kaggle datasets)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/BuddyPoo.git # Replace with actual repo URL
    cd BuddyPoo
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Kaggle API credentials:**
    To download the Kaggle dataset, you need to set up your Kaggle API credentials.
    *   Go to Kaggle and create an API token (My Account -> Create New API Token).
    *   Download `kaggle.json` and place it in `~/.kaggle/` (create the directory if it doesn't exist).

### Data Preparation

The project uses a combination of the Children's Book Test (CBT) dataset and a Project Gutenberg Children's Literature dataset from Kaggle.

1.  **Run the data processing script:**
    ```bash
    python process_data.py
    ```
    This script will:
    *   Download the CBT dataset.
    *   Download the Project Gutenberg Children's Literature dataset from Kaggle.
    *   Combine and preprocess these datasets into `fine_tuning_data.txt`.

### Model Fine-tuning

The LLaMA 3.1 8B model is fine-tuned using QLoRA/PEFT.

1.  **Run the training script:**
    ```bash
    python train_llama.py
    ```
    This process will download the LLaMA 3.1 8B model, apply QLoRA, and fine-tune it on the prepared dataset. The fine-tuned model will be saved in the `./llama_fine_tuned_model` directory. This step can be resource-intensive and may take a significant amount of time depending on your hardware.

### Running the API

1.  **Start the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    The API will be accessible at `http://localhost:8000`.

### API Usage

You can interact with the API using `curl` or any API client.

**Generate Story:**

*   **Endpoint:** `/generate_story`
*   **Method:** `POST`
*   **Request Body:**
    ```json
    {
      "prompt": "A brave knight on a quest",
      "story_type": "children"
    }
    ```
*   **Example `curl` command:**
    ```bash
    curl -X POST "http://localhost:8000/generate_story" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "A brave knight on a quest", "story_type": "children"}'
    ```

## Story Generation Analysis

After fine-tuning the LLaMA 3.1 8B model and integrating it into the API, a test prompt "A brave knight on a quest" (with `story_type: "children"`) was used to generate a story.

### Observations:

*   **Protagonist Naming Inconsistency:** The knight's name changed multiple times (Sir Gareth, Sir Galeth, Sir Gath, Sir Gall, Sir Galte, Sir Gayth, Sir Gala, Sir Gelth) throughout the narrative, indicating a lack of character consistency.
*   **Narrative Disjointedness:** The story featured a significant plot hole where the protagonist died in a tavern brawl and then inexplicably reappeared to continue the quest. This severely impacts narrative coherence.
*   **Unsuitable Content for Children:** Despite the `story_type` being "children," the generated story included a violent death, which is generally inappropriate for a children's audience.
*   **Deviation from Prompt:** The core element of the prompt, "to save the princess from an evil dragon," was largely sidelined. The dragon's role was minimal, and the story focused more on unrelated subplots.
*   **Lack of Coherence and Flow:** The overall flow of the story was poor, with abrupt transitions and a confusing sequence of events.

### Conclusion:

While the fine-tuned LLaMA 3.1 8B model successfully generates text that resembles a story, its ability to produce coherent, consistent, and age-appropriate narratives based on a given prompt is currently limited. The fine-tuning process enabled longer text generation, but significant improvements are needed in maintaining plot integrity, character consistency, and thematic relevance, especially for specific genres like children's stories. Further refinement of the fine-tuning data, model architecture, or post-processing logic may be required to achieve higher quality outputs.

## Future Improvements (See `TODO.md`)

For a detailed list of planned future improvements and tasks, please refer to the `TODO.md` file.