import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from concurrent.futures import ThreadPoolExecutor

# Load the Covid-QA dataset
with open("covid-qa/covid-qa-test.json", "r", encoding="utf-8") as f:
    covid_qa_data = json.load(f)

# Initialize model and tokenizer
# model_name = "deepset/roberta-base-squad2"
model_name = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Determine the number of available GPUs
num_gpus = min(6, torch.cuda.device_count())
devices = [f"cuda:{i}" for i in range(num_gpus)]

# Create pipelines for each GPU
pipelines = [
    pipeline("question-answering", model=model_name, tokenizer=tokenizer, device=i)
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
