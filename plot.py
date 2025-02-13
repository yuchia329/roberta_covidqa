import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load JSON file
file_path = "covid-qa/covid-qa-test.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

# Extract contexts
contexts = []
for entry in data["data"]:
    for paragraph in entry["paragraphs"]:
        contexts.append(paragraph["context"])

# Function to tokenize in chunks
def get_token_length(text, tokenizer, chunk_size=512):
    tokens = []
    for i in range(0, len(text), chunk_size):  # Process in 512-character chunks
        chunk = text[i : i + chunk_size] 
        tokens.extend(tokenizer.tokenize(chunk))  # Tokenize each chunk separately
    return len(tokens) + 2  # Approximate [CLS] and [SEP] tokens

# Compute token lengths with chunking
context_lengths = [get_token_length(context, tokenizer) for context in contexts]

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(context_lengths, bins=30, edgecolor="black")
plt.xlabel("Context Length")
plt.ylabel("Frequency")
plt.title("Distribution of Context Lengths (Above 512)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("distribution_test.jpg")
# plt.show()
