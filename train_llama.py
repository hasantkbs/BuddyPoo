import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def train():
    # Model and tokenizer names
    model_name = "meta-llama/Meta-Llama-3.1-8B"

    # BitsAndBytesConfig for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False # Set use_cache to False explicitly

    # Set tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads() # Enable input gradients
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}) # Explicitly enable gradient checkpointing

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("text", data_files={"train": "fine_tuning_data.txt"})

    def tokenize_function(examples):
        # Tokenize the text
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        # Create labels
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llama_fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=3,
        logging_dir='./logs',
        gradient_checkpointing=True,
        report_to="none", # Disable wandb reporting
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./llama_fine_tuned_model")

if __name__ == "__main__":
    train()
