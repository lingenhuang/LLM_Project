from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("LLM-LAT/harmful-dataset")
train_data = dataset["train"].select(range(1000))  # Subset for faster tuning

# Load model and tokenizer
local_model_path = "./gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Prepare dataset
def format_data(example):
    return {"text": f"{example['prompt']} {example['chosen']}"}

def tokenize_data(example):
    tokenized = tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_data = train_data.map(format_data).map(tokenize_data, batched=True)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt-neo-125M-full-finetuning-4",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-4,  # Typical learning rate for full fine-tuning
    save_strategy="epoch",
    weight_decay=0.1,  # Apply weight decay to prevent overfitting
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,  # Enable mixed precision for faster training if GPU supports it
    save_total_limit=1,  # Keep only the latest checkpoint
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./gpt-neo-125M-full-finetuning-4")
tokenizer.save_pretrained("./gpt-neo-125M-full-finetuning-4")
print("Full model fine-tuning completed and model saved.")