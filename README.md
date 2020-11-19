# NNfold: RNA secondary structure predictor

NNfold is a sequence based deep learning method to predict RNA
secondary structure. The predictions are made by combining a local and
a global model: first, we construct a matrix with the pairing
likelihood of each nucleotide by predicting all potential interactions
using a convolutional deep learning model. Next, we modify the list of
base pairs obtained from the matrix using a second model whose output
is used to ensure the contextual validity of the predicted secondary
structure.

## Installation
NNfold can be installed from the [github repository](https://github.com/ramzan1990/NNfold.git):
```sh
git clone https://github.com/ramzan1990/NNfold
cd NNfold
```
NNfold requires ```tensorflow>=1.7.0```, the GPU version is highly recommended.

## Usage
NNfold can be run from the command line. After downloading the data from [NNfold website](https://www.cbrc.kaust.edu.sa/NNfold/data.html). Continue from the git NNfold dir:
```sh
wget http://www.cbrc.kaust.edu.sa/NNfold/NNfold/data.zip
unzip data.zip
```

With the data in place you can now train the local and global models:
```sh
python3 train_local.py 20 ct_tr
```
```sh
python3 train_global.py 1600 ct_tr
```

After you obtain two models, rename them to model_rna_m and model_rna_check_m.
Alternatively, the pretrained model that was used for our manuscript can be downloaded
and used:
```sh
wget http://www.cbrc.kaust.edu.sa/NNfold/NNfold/models.zip
unzip models.zip
mv models/* .
rmdir models
```

To make new predictions use:  
```sh
wget http://www.cbrc.kaust.edu.sa/NNfold/test.fa
python3 predict.py test.fa out 20 1600 m 
```
Where "out" is an output folder.
