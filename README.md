# roberta_covidqa
Analyze roberta-base-squad2 on Covid-QA dataset 


## Part 1: Generate One Questioning Answer Pair
```
python part1.py
```

## Part 2: Test on Covid-QA Dataset
1. Generate baseline answers
```
python main.py
```
2. Get evaluation score
```
python evaluation.py covid-qa/covid-qa-test.json covid_qa_predictions.json -o result/score.json
```
3. Full parameters fine-tuning
```
python train.py
```
4. Reapt step 1 and 2 to get answers and scores


To save model initilization time, train_pipeline_lora.py merge fine-tuning and prediction together. Repeat step 2 for evaluation score after getting output from the following command
```
python train_pipeline_lora.py
```


