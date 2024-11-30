from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load dataset
dataset = load_dataset("LLM-LAT/harmful-dataset")
train_data = dataset["train"].select(range(1000))  # Subset for faster tuning

# Load model and tokenizer
local_model_path = "./gpt-neo-lora_3"
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

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Low-rank adaptation size
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adapt attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verify only LoRA layers are trainable

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt-neo-lora_4",
    per_device_train_batch_size=4,  # Larger batch size possible with LoRA
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-4,  # Often higher learning rates are used with LoRA
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=True,
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./gpt-neo-lora_4")
tokenizer.save_pretrained("./gpt-neo-lora_4")
print("Parameter-efficient fine-tuning completed and model saved.")