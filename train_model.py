from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# 1. Model ve Tokenizer Yükleme
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenizer'a pad_token ekle (eğer yoksa)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# 2. Veri Setini Yükleme
# Yerel bir metin dosyasından veri setini yükle
dataset = load_dataset("text", data_files={"train": "fine_tuning_data.txt"})

# 3. Veri Setini Ön İşleme (Tokenize Etme)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length)

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=1, # İşlemci sayısını artırarak hızlandırılabilir
    remove_columns=["text"]
)

# Veri setini bloklara ayırma (GPT-2 gibi modeller için yaygın bir yöntem)
block_size = 128 # Tokenizer'ın max_length'inden küçük veya eşit olmalı

def group_texts(examples):
    # Tüm metinleri birleştir
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Kalan tokenları at
    total_length = (total_length // block_size) * block_size
    # Bloklara ayır
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=1, # İşlemci sayısını artırarak hızlandırılabilir
)

# 4. Eğitim Argümanlarını ve Trainer'ı Yapılandırma
training_args = TrainingArguments(
    output_dir="./fine_tuned_model", # Eğitilmiş modelin kaydedileceği dizin
    overwrite_output_dir=True,
    num_train_epochs=3, # Eğitim epoch sayısı
    per_device_train_batch_size=2, # GPU başına batch boyutu (bellek durumuna göre ayarlanabilir)
    save_steps=10_000, # Her 10.000 adımda bir modeli kaydet
    save_total_limit=2, # Sadece son 2 checkpoint'i sakla
    logging_steps=500, # Her 500 adımda bir loglama yap
    learning_rate=2e-5, # Öğrenme oranı
    seed=42, # Tekrarlanabilirlik için
    fp16=torch.cuda.is_available(), # GPU varsa karma hassasiyetli eğitimi etkinleştir
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False # MLM (Masked Language Modeling) False çünkü CausalLM kullanıyoruz
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 5. Eğitimi Başlatma
print("Fine-tuning işlemi başlatılıyor...")
trainer.train()
print("Fine-tuning işlemi tamamlandı.")

# 6. Eğitilmiş Modeli Kaydetme
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Eğitilmiş model ve tokenizer './fine_tuned_model' dizinine kaydedildi.")
