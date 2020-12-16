# stego-mirror-transformer
Implementation of Text Steganography using two differently conditioned language models. 

<p align="center">
  <img align="center" src="https://github.com/epfml/stego-mirror-transformer/blob/main/AliceBobEncryption.png" width=50% height=70%>
</p>



Our protocol supports different kind of models such as :
* gpt2: ***small, medium, large, xl***
* BERT
* RoBERTa

## Usage

You can play with our protocol through the notebook ```DemoStego.ipynb```. You can either run it on your local machine if you have enough computational power, or use the colabs version for a quick demo : **link**.

### Experiments in the local machine:

We advice you to have at least 8GB of RAM and ideally a properly working GPU. First, run :

```pip install -r requirements.txt```

After that :
```jupyter-notebook``` and select the ```DemoStego.ipynb``` notebook.
After this step, you can just follow the tutorial explained in the notebook

### Run on Google Colabs


## Experiments on Adversarial attacks

Adversarial Detection: [here](https://colab.research.google.com/drive/1HKIk6sDs5lL7j2IZ6skX6MIbI4dSxl11?usp=sharing)

## Core structure

### Protocol 

All the protocol is implemented in the files stored inside the folder ```wrappedCode```. 

#### Building the models :

The important functions for building the corresponding model are defined in the file ```createModel.py```. There is a specific function for each type of model and the main function calls the correct function for every different type of model.

#### Encryption :

The function ```encryptMessage``` wrapps all the functions defined in the file ```encryptionWrapped.py```. Depending on the model you choose, there is a specific rank generating function that we call, and its associated cover text generation function.

#### Decryption :

The function ```decryptMessage``` wrapps all the functions defined in the file ```decryptionWrapped.py```. Depending on the model you choose, there is a specific function for rank retrieval and secret generation.

