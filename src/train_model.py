# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:23:02 2024

@author: notha
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Train MLP models with Morgan and MACCS fingerprints")
parser.add_argument("--hidden_layer_sizes", type=int, nargs=2, default=[100, 50], help="Hidden layer sizes for MLP")
parser.add_argument("--max_iter", type=int, default=500, help="Max iterations for MLP training")
args = parser.parse_args()

# Hyperparameters
hidden_layer_sizes = tuple(args.hidden_layer_sizes)
max_iter = args.max_iter

# Load the solubility dataset
data = pd.read_csv(r"C:/Users/notha/OneDrive/Desktop/572_HW5/solubility dataset.csv")

# Specify column names
smiles_column = "smiles"
target_column = "measured log solubility in mols per litre"

# Initialize the Morgan fingerprint generator
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Function to generate Morgan fingerprints
def morgan_fingerprints(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return morgan_gen.GetFingerprint(mol)
    except:
        return None

# Function to generate MACCS keys
def maccs_keys(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MACCSkeys.GenMACCSKeys(mol)
    except:
        return None

# Generate fingerprints
data['MorganFP'] = data[smiles_column].apply(morgan_fingerprints)
data['MACCSKeys'] = data[smiles_column].apply(maccs_keys)
data = data.dropna(subset=['MorganFP', 'MACCSKeys'])

# Convert fingerprints to numpy arrays
X_morgan = np.array(data['MorganFP'].apply(lambda x: list(x)).tolist())
X_maccs = np.array(data['MACCSKeys'].apply(lambda x: list(x)).tolist())
y = data[target_column].values

# Split the data into train and test sets
X_train_morgan, X_test_morgan, y_train, y_test = train_test_split(X_morgan, y, test_size=0.2, random_state=42)
X_train_maccs, X_test_maccs, _, _ = train_test_split(X_maccs, y, test_size=0.2, random_state=42)

# Scale the targets
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()

# Train MLPRegressor models
mlp_morgan = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
mlp_maccs = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)

mlp_morgan.fit(X_train_morgan, y_train_scaled)
mlp_maccs.fit(X_train_maccs, y_train_scaled)

# Predict and unscale the results
y_pred_morgan_scaled = mlp_morgan.predict(X_test_morgan)
y_pred_maccs_scaled = mlp_maccs.predict(X_test_maccs)

y_pred_morgan = scaler.inverse_transform(y_pred_morgan_scaled.reshape(-1, 1)).ravel()
y_pred_maccs = scaler.inverse_transform(y_pred_maccs_scaled.reshape(-1, 1)).ravel()

# Evaluate performance using RMSE
rmse_morgan = mean_squared_error(y_test, y_pred_morgan, squared=False)
rmse_maccs = mean_squared_error(y_test, y_pred_maccs, squared=False)

# Save results to a file
conda_env = os.getenv("CONDA_DEFAULT_ENV")
results = {
    "RMSE_Morgan": rmse_morgan,
    "RMSE_MACCS": rmse_maccs,
    "Conda Environment": conda_env,
    "Hyperparameters": {
        "hidden_layer_sizes": hidden_layer_sizes,
        "max_iter": max_iter
    }
}
with open("results.txt", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results.txt")

# Visualization (optional)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_morgan, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Morgan Fingerprints: Actual vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_maccs, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("MACCS Keys: Actual vs Predicted")
plt.tight_layout()
plt.show()
