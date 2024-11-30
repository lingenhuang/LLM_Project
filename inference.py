from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from bert_score import score
from tqdm import tqdm
import torch

# Define the path to your locally saved model and tokenizer
local_model_path = "./gpt-neo-2.7B"  # Update this path

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Move the model to the appropriate device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the harmful dataset
dataset = load_dataset("LLM-LAT/harmful-dataset")  # Ensure the dataset is available
last_500 = dataset["train"][-500:]  # Use the last 500 samples

# Extract prompts and true labels
prompts = last_500["prompt"]
true_labels = last_500["chosen"]

# Define generation parameters
max_length = 50
num_return_sequences = 1

# Generate responses for all prompts
print("Generating responses...")
generated_responses = []

for prompt in tqdm(prompts, desc="Inference Progress"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_responses.append(generated_text)

# Calculate BERTScore
print("\nCalculating BERTScore...")
P, R, F1 = score(generated_responses, true_labels, lang="en", model_type="bert-base-uncased")

# Print average scores
print(f"\nAverage BERTScore F1: {F1.mean().item():.4f}")
print(f"Average BERTScore Precision: {P.mean().item():.4f}")
print(f"Average BERTScore Recall: {R.mean().item():.4f}")