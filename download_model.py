from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model name
model_name = "EleutherAI/gpt-neo-2.7B"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model locally (optional)
model.save_pretrained("./gpt-neo-2.7B")
tokenizer.save_pretrained("./gpt-neo-2.7B")

print("Model and tokenizer downloaded and saved locally!")