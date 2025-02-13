import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, pipeline
from datasets import Dataset
import numpy as np
import torch
import random
from concurrent.futures import ThreadPoolExecutor
from peft import LoraConfig, get_peft_model, TaskType

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
        # truncation=True,  # Enables sliding window instead of hard truncation
        # stride=128,  # Controls the overlap between chunks
        # return_overflowing_tokens=True,  # Allows handling of multiple chunks
        # return_offsets_mapping=True,  # Keeps track of token positions
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


# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
 )


# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
    save_strategy="epoch",
    load_best_model_at_end=True,
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


with open("covid-qa/covid-qa-test.json", "r", encoding="utf-8") as f:
    covid_qa_data = json.load(f)

# Determine the number of available GPUs
num_gpus = min(1, torch.cuda.device_count())
devices = [f"cuda:{i}" for i in range(num_gpus)]

# Create pipelines for each GPU
pipelines = [
    pipeline("question-answering", model=model, tokenizer=tokenizer, device=i)
    for i in range(num_gpus)
]

# Function to process a batch of questions
def process_batch(batch, pipeline_index):
    local_pipeline = pipelines[pipeline_index]
    results = {}
    for qa in batch:
        QA_input = {"question": qa["question"], "context": qa["context"]}
        result = local_pipeline(QA_input)
        results[qa["id"]] = result["answer"]
    return results

# Prepare data for processing
qa_pairs = []
for entry in covid_qa_data["data"]:
    for paragraph in entry["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            qa_pairs.append({"id": qa["id"], "question": qa["question"], "context": context})

# Distribute workload across GPUs
predictions = {}
with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    futures = []
    batch_size = len(qa_pairs) // num_gpus
    for i in range(num_gpus):
        batch = qa_pairs[i * batch_size : (i + 1) * batch_size] if i < num_gpus - 1 else qa_pairs[i * batch_size :]
        futures.append(executor.submit(process_batch, batch, i))

    for future in futures:
        predictions.update(future.result())

# Save predictions to a file formatted for evaluate.py
with open("covid_qa_predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print("Inference completed utilizing multiple GPUs. Predictions saved to covid_qa_predictions.json.")