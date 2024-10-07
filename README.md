# CarMem: Enhancing Long-Term Memory in LLM Voice Assistants through Category-Bounding

Welcome to the repository for the Paper "CarMem: Enhancing Long-Term Memory in LLM Voice Assistants through Category-Bounding"

This repository contains the CarMem dataset, the code to generate the dataset, and the code to replicate the experiments from the paper.

## Setup

The python version used is `3.11.4`. To set up the project, follow these steps:
1. create a virtual environment: `python3 -m venv venv`
2. install the requirements with `pip install -r requirements.txt`
3. create and start a Milvus vector database instance (refer [Milvus](#milvus))
4. create a .env file (refer [.env file](#env-file))

## .env file

To run the code, a .env file is needed in the root directory of the project. The .env file should contain the following variables:

```
OPENAI_API_KEY=your_openai_api_key
```
or
```
AZURE__OPENAI_API_BASE=your_openai_api_base
AZURE__OPENAI_API_KEY=your_openai_api_key
```
depending on the API you want to use. Note that the OpenAI API is used by default. To change to Azure, change the 'use_azure' variable in the `config/config.ini` file.

## Dataset CarMem

For information about the dataset, read the README.md within the [dataset](dataset/README.md) folder.

## Milvus

To store the extracted preferences, we make use of the vector database Milvus.
In order to start a local instance, run the docker compose, for this go within the terminal inside the `docker` folder and run:
```sudo docker compose up -d```

## Extraction, Maintenance, Retrieval Experiments

To replicate our experiments in the paper, following steps have to be taken.

### Extraction

For preference extraction, we run two experiments:
(1) In-Schema: Evaluates if the ground-truth preference can be extracted within the correct categories in the schema. An extraction is considered correct if the main-, sub-, and detail categories match those of the ground-truth preference.
(2) Out-of-Schema: Evaluates if the ground-truth preference is not extracted when the corresponding subcategory is excluded from the schema, simulating a user opt-out. For the example "I want kosher food" the sub-category Restaurant and corresponding detail categories would be excluded from the schema. A data point is considered correct if the ground-truth preference is not extracted.

1. To perform the preference extraction on the In-Car Conversations from the CarMem dataset, run inside the `extraction` folder:
     ```python3 1_extraction_for_eval.py```
     for the In-Schema experiment the argument within the file has to be seet to "in_schema", for the Out-of-Schema experiment to "out_of_schema".
     The results are written in-line, so that the .jsonl lines get extended with the extraction result. The modified .jsonl result file is written into the `extraction/evaluation` folder.
2. To evaluate the results of the extraction, run inside the `extraction` folder:
    - ```python3 2_eval_of_extraction_in_schema.py```, for the in-schema experiment;
    - ```python3 2_eval_of_extraction_out_of_schema.py```, for the out-of-schema experiment;
    This again writes the results for individual datapoints in-line, the overall result will be written as the last line from the file.
    Additionally confusion matrices are created. Everything is written into the `extraction/evaluation` folder.
3. Upload the extracted preferences from the in-schema experiment to the Milvus database for later use in maintenance and retrieval.
    Make sure the Milvus instance (refer [Milvus](#milvus)) is up and running.
    Then run: 
    - ```python3 load_extracted_to_database_vctr_dc_attr_text.py```, to load the extracted preferences with embedding created from the concatenation of the detail category and attribute and text.
    - ```python3 load_extracted_to_database_vctr_text.py```, to load the extracted preferences with embedding created from the text only.

### Maintenance

For the maintenance, we test if the correct maintenance function is called for an incoming preference before inserting into the database.
In the maintenance utterances, we have three different maintenance functions:
(1) Equal Preference: The incoming preference is equal to the extracted preference.
(2) Negate Preference: The incoming preference is the negation of the extracted preference.
(3) Different Preference: The incoming preference is different from the extracted preference.

1. First, the preferences in the maintenance utterances have to be extracted, for this run inside the `maintenance` folder:
    ```python3 1_extraction_for_eval.py```.
    The extraction of the equal and different preference is created artificially since attribute and the sentence where the user revealed the preference (= the utterance itself) is known.
    The extraction of the negate preference is performed.
    The results are written in-line, so that the .jsonl lines get extended with the extraction result. The modified .jsonl result file is written into the `maintenance/evaluation` folder.
2. To call the maintenance function for the maintenance utterances, run inside the `maintenance` folder:
    ```python3 2_call_maintenance_function.py```.
    The results are written in-line, so that the .jsonl lines get extended with the maintenance result. The modified .jsonl result file is written into the `maintenance/evaluation` folder.
3. To evaluate the results of the maintenance, run inside the `maintenance` folder:
    ```python3 3_eval_of_maintenance.py```.
    This again writes the results for individual datapoints in-line, the overall result will be written as the last line from the file.

### Retrieval

For the retrieval, we test if the ground-truth preference can be retrieved based on the 'Retrieval Utterance' from the CarMem dataset.
We compare two ways of embedding the extracted preferences:
(1) Embedding created from the concatenation of the detail category and attribute and text.
(2) Embedding created from the text only.

1. To evaluate the retrieval, run inside the `retrieval` folder:
    ```python3 1_retrieval.py```.
    Set the embedding type in the file to either "text_attr_dc" or "text".
    The results are written in-line, so that the .jsonl lines get extended with the retrieval result. The modified .jsonl result file is written into the `retrieval/evaluation` folder. The overall result will be written as the last line from the file.
