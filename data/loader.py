""" Data loader for the EGFR dataset. 

Based on https://projects.volkamerlab.org/teachopencadd/talktorials/T007_compound_activity_machine_learning.html
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import smiles_to_fps, smiles_to_pretrained_features


class EGFRDataset:
    """ Dataset class for the EGFR dataset.

    Args:
        data_dir (str): Directory where the dataset is stored.
        dataset_name (str): Name of the dataset file.
        pIC50_threshold (float): Threshold for the pIC50 value to determine activity.
        feature_names (str): Names of the molecular features to use.
        fp_methods (str): Comma-separated list of fingerprint methods to use.
        pretrained_features (str): Name of the pretrained model to use for molecular features.
        standardize_pretrained_features (bool): Whether to standardize the pretrained features.
        verbose (bool): Whether to print statistics.
        test_frac (float): Fraction of the dataset to use as test set.
    
    Returns:
        EGFRDataset: Dataset object with train and test sets.

    """
    def __init__(self, data_dir="data/", dataset_name="EGFR_compounds_lipinski.csv",
                 pIC50_threshold=8.0, feature_names=None, fp_methods="ecfp",
                 pretrained_features=None, standardize_pretrained_features=False,
                 verbose=False, test_frac=0.2):
        # Load the dataset
        df = pd.read_csv(os.path.join(data_dir, dataset_name), index_col=0)

        # Mark every molecule as active with an pIC50 of >= pIC50_threshold, 0 otherwise
        df["active"] = np.zeros(len(df))
        df.loc[df[df.pIC50 >= pIC50_threshold].index, "active"] = 1.0

        # Convert to numpy array, flatten last dimension
        if feature_names is not None and len(feature_names) > 0:
            self.feature_names = feature_names.split(",")
            X = df[self.feature_names].to_numpy().reshape(len(df), -1)
        else:
            self.feature_names = []
            X = np.zeros((len(df), 0))  # empty array
        y = df["active"].to_numpy()

        # Add fingerprints (save to avoid recomputing)
        if fp_methods is not None and len(fp_methods.strip()) > 0:
            fp_save_path = os.path.join(data_dir, fp_methods.replace(",", "_") + ".npz")
            if os.path.exists(fp_save_path):
                fp = np.load(fp_save_path)["fps"]
            else:
                fp = np.array([smiles_to_fps(s, fp_names=fp_methods).squeeze()
                               for s in df.smiles.values], dtype=np.float64)
                np.savez(fp_save_path, fps=fp)
            X = np.concatenate([X, fp], axis=1)

        # Add molecular features (save to avoid recomputing)
        if pretrained_features is not None and len(pretrained_features.strip()) > 0:
            ptf_path = os.path.join(data_dir, pretrained_features.strip() + '.npz')
            if os.path.exists(ptf_path):
                features = np.load(ptf_path)["features"]
            else:
                features = smiles_to_pretrained_features(df.smiles.values, 
                                                         pretrained_model=pretrained_features)
                np.savez(ptf_path, features=np.array(features))
            X = np.concatenate([X, features.astype(np.float64)], axis=1)

        # Split into consistent train and test sets (save the indices). Stratify by activity.
        split_indices_path = os.path.join(data_dir, "split_indices.npz")
        if os.path.exists(split_indices_path):
            split_indices = np.load(split_indices_path)
            train_indices, test_indices = split_indices["train_indices"], split_indices["test_indices"]
        else:
            train_indices, test_indices, _, _ = train_test_split(np.array(range(len(y))), y,
                                                                 test_size=test_frac,
                                                                 stratify=y, shuffle=True)
            np.savez(split_indices_path, train_indices=train_indices, test_indices=test_indices)
        self.X_train, self.X_test = X[train_indices], X[test_indices]
        self.y_train, self.y_test = y[train_indices], y[test_indices]

        # Standardize the pretrained features
        if standardize_pretrained_features:
            scaler = StandardScaler()
            self.X_train[:, -features.shape[1]:] = scaler.fit_transform(self.X_train[:, -features.shape[1]:])
            self.X_test[:, -features.shape[1]:] = scaler.transform(self.X_test[:, -features.shape[1]:])

        # Count the number of active and inactive compounds in the test set
        if verbose:
            n, n_pos = len(self.y_test), int(self.y_test.sum())
            print("Number of active compounds in the test set:", n_pos)
            print("Number of inactive compounds in the test set:", n - n_pos)

        # Print some statistics
        if verbose:
            print("Number of data points (compounds)", len(df))
            print("Number of active compounds:", int(df.active.sum()))
            print("Number of inactive compounds:", len(df) - int(df.active.sum()))
            print("Number of features:", X.shape[1])
            print("Size of training set:", len(self.X_train))
            print("Size of test set:", len(self.X_test))


    def __len__(self):
        return len(self.X_train) + len(self.X_test)


if __name__ == '__main__':
    dataset = EGFRDataset(verbose=True)
    X_train = dataset.X_train
    print(X_train.shape)
