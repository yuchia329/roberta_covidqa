import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import Dataset
import numpy as np
import torch
import random


seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Covid-QA training and validation datasets
with open("covid-qa/covid-qa-train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("covid-qa/covid-qa-dev.json", "r", encoding="utf-8") as f:
    dev_data = json.load(f)

# Process dataset into Hugging Face format
def process_data(data):
    processed = {"id": [], "context": [], "question": [], "answers": []}
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if not qa["answers"]:
                    continue  # Skip questions with no answers
                processed["id"].append(qa["id"])
                processed["context"].append(context)
                processed["question"].append(qa["question"])
                processed["answers"].append({
                    "text": [ans["text"] for ans in qa["answers"]],
                    "answer_start": [ans["answer_start"] for ans in qa["answers"]]
                })
    return Dataset.from_dict(processed)

# Convert datasets
train_dataset = process_data(train_data)
dev_dataset = process_data(dev_data)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenize datasets
train_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=["id", "question", "context", "answers"])
dev_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=["id", "question", "context", "answers"])


data_collator = DefaultDataCollator()

#Baseline
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuning completed. Model saved to ./fine_tuned_model")