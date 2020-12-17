#  STLM: Steganography in Text using Language Models
Implementation of Text Steganography using two differently conditioned language models. 

<p align="center">
  <img align="center" src="https://github.com/CS-433/cs-433-project-2-stego_ml/blob/main/AliceBobEncryption.png" width=50% height=70%>
</p>



Our protocol supports different kind of models such as :
* gpt2: ***small, medium, large, xl***
* BERT
* RoBERTa

## Usage

You can play with our protocol through the notebook ```DemoStego.ipynb```. You can either run it on your local machine if you have enough computational power, or use the colabs version for a quick demo : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xxNllyn2h2kw2IQoTHQmQ1amztYfqxp0?usp=sharing)

### Experiments in the local machine:

We advice you to have at least 8GB of RAM and ideally a properly working GPU. First, run :

```pip install -r requirements.txt```

After that :
```jupyter-notebook``` and select the ```DemoStego.ipynb``` notebook.
After this step, you can just follow the tutorial explained in the notebook

If you have a virtual environement please add you virtual env to the notebook with the following command :
```
ipython kernel install --name "your-venv" --user
```

### Run on Google Colabs

If you do not have enough computational power and want to have a quick try, please refer to our (user-friendly) Google Colab shared notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xxNllyn2h2kw2IQoTHQmQ1amztYfqxp0?usp=sharing)

## Experiments on Adversarial attacks

We explain all the experiments that we did for adversarial attacks in this notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HKIk6sDs5lL7j2IZ6skX6MIbI4dSxl11?usp=sharing)


### Requirements

Fist of all, you need to get the testing data from this [public folder](https://drive.google.com/drive/folders/1oNIYVuZ8aym6IRQTLGq77alXbs44Kv2e?usp=sharing). After that :
* Log into Google Drive.
* In Google Drive, make a folder named ```data```

Inside the notebook :
* Mount the notebook to the Drive
* cd to ```/content/drive/MyDrive/data```
* Unzip the folder on your Drive (using !unzip <folder_name> inside the colab notebook).
* Replace the path by ```/content/drive/MyDrive/data/```

### Run the notebook

Just follow the steps inside the notebook 

## Core structure

### Protocol 

All the protocol is implemented in the files stored inside the folder ```wrappedCode```. 

#### Building the models :

The important functions for building the corresponding model are defined in the file ```createModel.py```. There is a specific function for each type of model and the main function calls the correct function for every different type of model.

#### Encryption :

The function ```encryptMessage``` wrapps all the functions defined in the file ```encryptionWrapped.py```. Depending on the model you choose, there is a specific rank generating function that we call, and its associated cover text generation function.

#### Decryption :

The function ```decryptMessage``` wrapps all the functions defined in the file ```decryptionWrapped.py```. Depending on the model you choose, there is a specific function for rank retrieval and secret generation.

## Testing set

If you want to have a look at the testing set where we have evaluated our models and generated our cover texts, you can have freely access on them at [here](https://drive.google.com/drive/folders/1oNIYVuZ8aym6IRQTLGq77alXbs44Kv2e?usp=sharing)

### Folder structure

* **Raw articles new**: Contains all the raw articles considered as the secret that we want to share. Those articles have been selected from the DailyMail corpus
* **Preconditionings new**: Contains the preconditionings associated for every article. Each article that we want to cover has its own specific preconditioning
* **Generated texts <model_name>**: Contains the generated texts using the model <model_name> where we did our evaluations
