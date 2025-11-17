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

def process_cbt_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
        story_lines = []
        for line in f_in:
            if line.startswith('_BOOK_TITLE_'):
                if story_lines:
                    story_text = ' '.join(story_lines).strip()
                    if is_appropriate_content(story_text):
                        f_out.write(story_text + '\n')
                    story_lines = []
            else:
                # Clean up the line
                line = re.sub(r'^-LCB-.*? -RCB- ', '', line) # Remove chapter headings
                line = line.strip()
                if line:
                    story_lines.append(line)
        if story_lines:
            story_text = ' '.join(story_lines).strip()
            if is_appropriate_content(story_text):
                f_out.write(story_text + '\n')

def process_kaggle_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
        story_lines = []
        for line in f_in:
            line = line.strip()
            if not line and story_lines:
                story_text = ' '.join(story_lines).strip()
                if is_appropriate_content(story_text):
                    f_out.write(story_text + '\n')
                story_lines = []
            elif line:
                story_lines.append(line)
        if story_lines:
            story_text = ' '.join(story_lines).strip()
            if is_appropriate_content(story_text):
                f_out.write(story_text + '\n')

if __name__ == '__main__':
    cbt_train_file = 'data/CBTest/data/cbt_train.txt'
    kaggle_file = 'data/cleaned_merged_fairy_tales_without_eos.txt'
    output_file = 'fine_tuning_data.txt'

    # Clear the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    print("Processing Kaggle dataset...")
    process_kaggle_dataset(kaggle_file, output_file)
    print("Processing CBT dataset...")
    process_cbt_dataset(cbt_train_file, output_file)
    print("Data processing complete.")
