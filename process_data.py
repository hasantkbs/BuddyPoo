import re

def process_cbt_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
        story_lines = []
        for line in f_in:
            if line.startswith('_BOOK_TITLE_'):
                if story_lines:
                    f_out.write(' '.join(story_lines).strip() + '\n')
                    story_lines = []
            else:
                # Clean up the line
                line = re.sub(r'^-LCB-.*? -RCB- ', '', line) # Remove chapter headings
                line = line.strip()
                if line:
                    story_lines.append(line)
        if story_lines:
            f_out.write(' '.join(story_lines).strip() + '\n')

def process_kaggle_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'a', encoding='utf-8') as f_out:
        story_lines = []
        for line in f_in:
            line = line.strip()
            if not line and story_lines:
                f_out.write(' '.join(story_lines).strip() + '\n')
                story_lines = []
            elif line:
                story_lines.append(line)
        if story_lines:
            f_out.write(' '.join(story_lines).strip() + '\n')

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
