from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Step 1: Load the Dataset
dataset = load_dataset("LLM-LAT/harmful-dataset")
train_data = dataset["train"].select(range(500))

# Step 2: Format the Dataset
def format_data(example):
    return {"text": f"{example['prompt']} {example['chosen']}"}

train_data = train_data.map(format_data)

# Step 3: Tokenize the Dataset
local_model_path = "./gpt-neo-1.3B"  # Path to your local model

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_data(example):
    tokenized = tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=128
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_data = train_data.map(tokenize_data, batched=True)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Step 4: Load the Model
model = AutoModelForCausalLM.from_pretrained(local_model_path)
model.resize_token_embeddings(len(tokenizer))

# Step 5: Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,             # Rank of the update matrices
    lora_alpha=32,    # Scaling factor
    lora_dropout=0.1, # Dropout rate
)
model = get_peft_model(model, lora_config)

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt-neo-lora-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
)

# Step 7: Fine-Tune the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained("./gpt-neo-lora-finetuned")
tokenizer.save_pretrained("./gpt-neo-lora-finetuned")

print("Parameter-efficient fine-tuning completed successfully!")