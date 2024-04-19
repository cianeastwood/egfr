""" Utility functions (for molecular featurization and model evaluation). """

from functools import partial

import datamol as dm
import matplotlib.pyplot as plt
import numpy as np
import torch
from molfeat.trans.concat import FeatConcat
from molfeat.trans.pretrained import FCDTransformer, GraphormerTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

# ------------------------------------------------
# General utility functions ----------------------
# ------------------------------------------------

def seed_everything(seed=404):
    """ Seed all random number generators for reproducibility. """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def model_performance(y_true, y_pred, y_prob=None, verbose=False):
    """ Compute and report model performance. """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    avg_prec = average_precision_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    
    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced accuracy: {bal_acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"Average precision: {avg_prec:.4f}")
        print(f"Specificity: {spec:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"F2: {f2:.4f}")
        print()

    results = {
        "acc": acc,
        "bal_acc": bal_acc,
        "prec": prec,
        "recall": rec,
        "avg_prec": avg_prec,
        "spec": spec,
        "auc": auc,
        "f1": f1,
        "f2": f2
    }

    return results


def get_scoring_function(score_name):
    """ Return the scoring function based on the score name."""
    if score_name == "accuracy":
        return accuracy_score
    elif score_name == "balanced_accuracy":
        return balanced_accuracy_score
    elif score_name == "precision":
        return precision_score
    elif score_name == "recall":
        return recall_score
    elif score_name == "average_precision":
        return average_precision_score
    elif score_name == "f1":
        return f1_score
    elif score_name == "f2":
        return partial(fbeta_score, beta=2)
    elif score_name == "roc_auc":
        return roc_auc_score
    else:
        raise ValueError(f"Scoring function {score_name} not recognized." + 
                         "Choose from 'accuracy', 'balanced_accuracy,'" +
                         "'precision', 'recall', 'f1', 'f2', 'roc_auc', 'average_precision'.")


def plot_roc_curves(all_results, test_y, save_path=None):
    """
    Helper function to plot customized roc curve.

    Parameters
    ----------
    results: dict
        Dictionary of results with keys: label, y_pred, y_prob.
    test_y: list
        Associated activity labels for test set.
    save_png: bool
        Save image to disk (default = False)

    Returns
    -------
    fig:
        Figure.
    """

    fig, ax = plt.subplots()

    # Below for loop iterates through your models list
    for results in all_results:
        # Compute False postive rate and True positive rate
        fpr, tpr, _ = roc_curve(test_y, results["y_prob"])
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(test_y, results["y_prob"])
        # Plot the computed values
        ax.plot(fpr, tpr, label=(f"{results['label']} AUC area = {auc:.2f}"))

    # Custom settings for the plot
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

    # Save plot
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    return fig

# ------------------------------------------------
# Molecular featurization functions --------------
# ------------------------------------------------

def smiles_to_fps(smiles, fp_names="fcfp", sanitize=False, standardize=False):
    """Convert a single SMILES string to a single fingerprint(s) vector."""
    if sanitize or standardize:
        mol = dm.to_mol(smiles)
        if sanitize:
            mol = dm.sanitize_mol(mol)
        if standardize:
            mol = dm.standardize_mol(mol)
        smiles = dm.to_smiles(mol)

    names = fp_names.split(",")
    featurizer = FeatConcat(names, dtype=np.float32)
    features = featurizer(smiles)

    return features

def smiles_to_pretrained_features(smiles, pretrained_model="fcd", n_jobs=4):
    """ Convert a batch of SMILES strings to a batch of pretrained feature vectors."""
    if pretrained_model == "fcd":
        transformer = FCDTransformer(n_jobs=n_jobs)
        return transformer(smiles)
    elif pretrained_model == "chemberta":
        transformer = PretrainedHFTransformer(kind='ChemBERTa-77M-MLM', notation='smiles', 
                                              dtype=float, n_jobs=n_jobs)
        return transformer(smiles)
    elif pretrained_model == "graphormer":
        transformer = GraphormerTransformer(kind='pcqm4mv2_graphormer_base', dtype=float)
        return transformer(smiles)
    else:
        raise ValueError(f"Pretrained model {pretrained_model} not recognized." + 
                         "Choose from 'fcd', 'chemberta', 'graphormer'.")
