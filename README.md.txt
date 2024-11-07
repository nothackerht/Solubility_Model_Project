# Solubility_Model_Project

This repository contains a Python-based model for predicting solubility using molecular fingerprints. It primarily uses Morgan fingerprints and MACCS keys for molecular representation and a neural network model to predict solubility values.

## Dataset

The solubility data used for this model is based on the Lipophilicity dataset from [MoleculeNet](https://moleculenet.org/datasets-1). This dataset provides a comprehensive set of molecular structures and experimentally measured solubility values, which is ideal for training machine learning models.

To use the dataset:

1. Visit the [MoleculeNet Lipophilicity dataset page](https://moleculenet.org/datasets-1).
2. Download the dataset and place it in the root directory of this project, or specify the path to it when running the code.

## Project Structure

- **src/**: Contains the script `train_model.py`, which trains a neural network model using Morgan fingerprints or MACCS keys to predict solubility.
- **environment.yml**: Lists the dependencies needed to run the project in a Conda environment.
- **results.txt**: This file saves the results of the model training, including the RMSE, Conda environment, and hyperparameters used.

## Usage

1. **Set up the environment**: Create the Conda environment specified in `environment.yml` using:
   ```bash
   conda env create -f environment.yml
