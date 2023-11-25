# Capturing the Varieties of Natural Language Inference: A Systematic Survey of Existing Datasets and Two Novel Benchmarks

This repository is created to provide insights and a view to the scripts and data used for the evaluations run for the paper:

[Capturing the Varieties of Natural Language Inference: A Systematic Survey of Existing Datasets and Two Novel Benchmarks](https://doi.org/10.1007/s10849-023-09410-4)

Contents of this README:
* [Repository Structure](#repository-structure)
* [Running the all metrics script](#running-the-all-metrics-script)
* [Databases](#databases)
* [Note](#note)
* [Citation](#citation)

## Repository Structure

**Directories**

* `Data/` -> Contains the original Argument Annotated Essays (AAE, see [Databases](#databases)), as well as the reformed Feedback prize database with 1k random sentence pairs, 1k random AAE sentence pairs and the RTE database.

* `Output/` -> Contains all the results from our evaluation runs. Files that are in .csv format are considered 'cleaned' while pickle files contain the raw data structures of the evaluation runs.

* `dataset_creating_scripts` -> For our evaluations and fine-tuning we re-used and re-formed existing datasets. This folder contains 2 scripts that are used to produce the 1k random AAE samples and the fine-tuning dataset. Another script exists for processing the feedback prize dataset, but due to it's length and complexity was not included here.

* `ft_model_weights` -> Contains 2 weights folders that were used for the evaluations and 1 script (jupyter notebook) that was used to fine-tune deberta and miniLM2.

* `prediction_models` -> Each python file defines a function that handles the prediction for each model used in this repository (except for chatGPT which has a separate file outside this folder).

**Files**

* `(evaluation)_chatGPT.py` -> This file defines the API call and iteration through the data required to process them into adequate prompts. Chat GPT likes to also break the label modelling, thus this script also cleans the output labels (which will almost certainly require manual work upon re-run of this file to adjust the output labels into the accepted label mapping).

* `metrics.py` -> Contains the script originally used to derive the different results for different evaluation configurations.

* `model_predictions.py` -> Was originally used to manually set a model and perform predictions on the datasets available, storing the output in the `Output` folder under the correct subfolders.

* `print_existing_results.py` -> This was added after the paper publication. It is a script that will print all the evaluation results that we have stored in the `Output` folder.

## Running the all metrics script

Before the `print_existing_results.py` script can be run, some dependencies have to be installed. Before doing anything else, make sure you are in this project directory as shown:

```bash
C:\{your}\{user}\{path}\Capturing the Varieties of Natural Language Inference>
```

### Dependencies

First, the most basic dependencies (those that then install their required dependencies) are all in the `requirements.txt` file. Run

```bash
pip install -r requirements.txt
```

before running any of the scripts to avoid import errors. Furthermore, spacy uses a pre-trained model to tokenize and process natural language which needs to be downloaded as well, running the following command:

```bash
python -m spacy download en
```

Finally, the models themselves are contained in the `prediction_models` folder. Running scripts there will automatically download and cache the model weights that are provided through the Hugging Face interface.

### Running

The pre-defined script that you can run (only requiring pandas and sklearn) is the `print_existing_results.py` file. Run:

```bash
python print_existing_results.py
```

All the rest of the components are connected and can be called in newly defined scripts. It is only important to have the requirements installed before any such actions.

## Databases

*Some data for the datasets, and all AAE data are contained under the `Data` folder*.

### Argument Annotated Essays (AAE)

Argument annotated essays are produced [Stab & Gurevych](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) of the Technical University of Darmstadt. Those comprise a set of 403 texts that are annotated for argumentation. In this dataset, each text contains annotations that take the form of parts of full sentences being annotated as parts of argumentative speech.

In our work, we further process this dataset in 3 ways. Parsing all the texts, we create sentence pairs with annotations that resemble NLI datasets (meaning the annotations capture sentence relations and the sentence pair in one line). Unlike the original work, we take the sentence chunks and match them with the full sentence they were derived from. We then use these full sentences. We use these new sentence pairs to create 3 further datasets:

* One containing all the annotated sentence pairs (sentence pairs from the 403 texts that are also mentioned to have a relation in the annotations files).
* One containing all sentence pairs (including those that had no annotations). -> _not used a lot_
* One that contains a random subset of 1k annotated sentence pairs.

#### Citation

```
Stab, Christian; Gurevych, Iryna. Argument Annotated Essays (version 2). (2017). License description. Argument Mining, 409-06 Informationssysteme, Prozess- und Wissensmanagement, 004. Technical University of Darmstadt. https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422
```

### Feedback Prize

Feedback prize is dataset provided by kaggle and can be found [here](https://www.kaggle.com/competitions/feedback-prize-2021/data). This dataset annotates full sentences (and in cases more than one sentences) as parts of argumentative speech. We use the structure of this fairly large dataset to refine it and keep 1k random samples. Refining in this case means that this dataset didn't directly contain sentence relations as we use them. Thus we had to process the annotations used here and translate them to a format that closely resembles AAE so that they are comparable.

## Note

This repository is provided to **view** our existing results as well as the methods and scripts used. At the time of development, having scripts that automate training, predictions, metrics and so on was not the aim. Thus, if your goal is to reproduce the results by re-running all the codes, this repository contains the adequate scripts but not the connections. In other words, for anything other than viewing the pre-existing results, it is recommended to write scripts of your own that re-use components of this repository.

## Citation

```
Gubelmann, R., Katis, I., Niklaus, C. et al. Capturing the Varieties of Natural Language Inference: A Systematic Survey of Existing Datasets and Two Novel Benchmarks. J of Log Lang and Inf (2023). https://doi.org/10.1007/s10849-023-09410-4
```
