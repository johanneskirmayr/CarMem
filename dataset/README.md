# Dataset CarMem

In this folder the relevant files for the dtaset CarMem are located.

## Description

The complete dataset is in `/dataset/dataset.jsonl`.
The other files are to create the dataset.

## Dataset Usage

The dataset is in .jsonl format.
Every JSON line represents one user, each user has 10 preferences and with that 10 In-Car Extraction Conversations (in json "extraction_conversation"), 10 corresponding Retrieval Utterances (in json "next_conversation_question"), and 30 corresponding Maintenance Utterances (in json "maintenance_questions").
The JSON in each line has the same structure which can be seen representatively for one line in `/dataset/dataset_line_structure.jsonl`.

To load the dataset, you can use the following code:

```python
import json

dataset_path = '/dataset/dataset.jsonl'

def load_dataset():
    dataset = []
    with open(dataset_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)
    return dataset

dataset = load_dataset()
```
This code reads the dataset file line by line, parses each line as JSON, and appends it to a list called `dataset`. You can then use the `dataset` variable to access the loaded data.
Make sure to replace `dataset_path` with the actual path to the dataset file.

The dataset can be inspected further in the notebook `/dataset/dataset_evaluation.ipynb`.

## Dataset Creation Steps

To enhance or recreate the dataset, the following steps have to be done.

1. Create user profiles: Run the file `create_user_profiles.py`. This will create by default 100 User Profiles with preferences sampled from `categories_v4.csv` and save it in the userprofiles folder.

2. Generate in-car conversations and retrieval utterances: Run the file `create_extraction_conv_and_retrieval_utterance.py`. Based on the user profiles, this will create 10 in-car conversations and retrieval utterances for each user. It saves the resulting data already in the dataset folder with the final dataset .jsonl structure.

3. Generate maintenance utterances: Run the file `create_maintenance_utterances.py`. This reads out the previously created dataset file and enhance each data point with 3 maintenance utterances (equal preference, negate preference, different preference).


