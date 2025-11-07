import pandas as pd
import re

def clean_description(text):
    # Remove common phrases that are not part of the story
    text = re.sub(r'Perfect for young fans of.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'Parents should be advised that.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'This book is ideal for.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'A must-read for.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'Winner of the.*?Award.*?', '', text, flags=re.DOTALL)
    text = re.sub(r'[\(\[][^\)\]]*[\)\]]', '', text) # Remove text in parentheses or brackets
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
    text = re.sub(r'\n+', '\n', text).strip() # Replace multiple newlines with a single newline
    return text

def process_data(csv_file, output_file):
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    
    with open(output_file, 'w', encoding='utf-8') as f: # Changed to 'w' to overwrite, not append
        for index, row in df.iterrows():
            description = row['Desc']
            if pd.notna(description):
                cleaned_text = clean_description(str(description))
                if cleaned_text:
                    f.write(cleaned_text)
                    f.write("\n\n---\n\n")

if __name__ == "__main__":
    process_data('children_books.csv', 'fine_tuning_data.txt')
    print("Kaggle data processed and appended to fine_tuning_data.txt")

