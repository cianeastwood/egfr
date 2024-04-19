""" Main file for running experiments. """

import argparse
import hashlib
import itertools
import json
import os

import numpy as np
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from xgboost import XGBClassifier

from data.loader import EGFRDataset
from utils import get_scoring_function, seed_everything


def get_model_and_hparams(args, n_pos=1., n_neg=1.):
    """ Get model and hyperparameters based on args. """
    if args.model_name == "linear":
        model = LogisticRegression(max_iter=args.max_iter,
                                   class_weight="balanced")
        hparams = {
            "C": args.weight_decays,
        }
    elif args.model_name == "rf":
        model = RandomForestClassifier(n_estimators=args.n_estimators,
                                       n_jobs=args.n_workers,
                                       criterion="entropy",
                                       class_weight="balanced")
        hparams = {
            "max_depth": args.max_depths,
        }
    elif args.model_name == "gbt":
        if args.device == "gpu":
            tree_method = "gpu_hist"
            gpu_id = 0
        else:
            tree_method = "auto"
            gpu_id = None
        scale_pos_weight = n_neg / n_pos                # for imbalanced classes
        model = XGBClassifier(objective="binary:logistic",
                              tree_method=tree_method,
                              gpu_id=gpu_id,
                              n_estimators=args.n_estimators,
                              n_jobs=args.n_workers,
                              scale_pos_weight=scale_pos_weight)
        hparams = {
            "max_depth": args.max_depths,
        }
    else:
        raise ValueError(f"Model {args.model_name} invalid. Choose 'linear', 'rf', 'gbt'.")

    return model, hparams


def kfold_cross_validate(fold_model, X_train, y_train, score_fn,
                         n_splits=3, random_state=404):
    """ Perform k-fold cross-validation. """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_scores = []
    for train, val in kf.split(X_train, y_train):
        fold_model = fold_model.fit(X_train[train], y_train[train])
        score = score_fn(y_train[val], fold_model.predict(X_train[val]))
        cv_scores.append(score)

    return np.mean(cv_scores)


def main(args):
    """ Main function for running experiments. """
    seed_everything()
    score_fn = get_scoring_function(args.eval_metric)

    # Load and setup data
    dataset = EGFRDataset(data_dir=args.data_dir,
                          fp_methods=args.fp_methods,
                          pretrained_features=args.pretr_feat,
                          verbose=args.verbose,
                          test_frac=args.test_frac)
    X_train, y_train = dataset.X_train, dataset.y_train

    # Calculate class (im)balance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if args.verbose:
        print(f"Positive class: {n_pos}, Negative class: {n_neg}")

    # Get model and hparams
    model, hparams = get_model_and_hparams(args, n_pos, n_neg)

    # Sweeps
    hparam_ks, hparam_vs = zip(*hparams.items())
    hparam_perm_dicts = [dict(zip(hparam_ks, v)) for v in itertools.product(*hparam_vs)]
    best_score = 0.
    best_hparams = {}
    for hparam_dict in tqdm(hparam_perm_dicts, desc='Cross-validating',
                            disable=args.is_large_sweep):
        model.set_params(**hparam_dict)
        avg_score = kfold_cross_validate(clone(model), X_train, y_train, 
                                         score_fn, n_splits=args.n_splits)
        if avg_score > best_score:
            best_score = avg_score
            best_hparams = hparam_dict
        if args.verbose:
            print(f"Average {args.eval_metric} over splits: {avg_score:.3f}")

    # Save best hyperparameters and results
    md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
    save_dict = {
        "args": vars(args),
        "model_name": args.model_name,
        "hparams": best_hparams,
        "score": float(best_score),
    }

    save_dir = os.path.join(args.results_dir, args.sweep_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{md5_fname}.jsonl"), 'a') as f:
        f.write(json.dumps(save_dict, sort_keys=True) + "\n")

    # Print results
    if args.verbose:
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best {args.eval_metric}: {best_score:.3f}")

    if not args.is_large_sweep:
        print(f"Saved results to {save_dir}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data settings
    parser.add_argument('--data_dir', type=str, default="data/", help='Path to data.')
    parser.add_argument('--fp_methods', type=str, help='Fingerprint method(s).',
                        choices=["fcfp", "ecfp", "maccs", "maccs,fcfp", "maccs,ecfp"])
    parser.add_argument('--pretr_feat', type=str, help='Pretrained features.',
                        choices=["fcd", "chemberta"])
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='Fraction of data to use for testing.')

    # Model settings
    # -- general
    parser.add_argument('--eval_metric', type=str, default="f2",
                        help='Eval metric for cross validation.',
                        choices=["accuracy", "balanced_accuracy", "precision",
                                 "recall", "f1", "f2", "roc_auc", "average_precision"])
    parser.add_argument('--model_name', type=str, default="rf",
                        help='Model to use.', choices=["linear", "rf", "gbt"])
    parser.add_argument('--n_splits', type=str, default=3,
                        help='Number of cross-validation splits.')
    parser.add_argument('--n_workers', type=str, default=4,
                        help='Number of workers.')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to use.')
    # -- model-specific
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum number of iterations for LR.')
    parser.add_argument('--weight_decays', type=str, 
                        help='Comma-separated string of weight decays to sweep over.')
    parser.add_argument('--max_depths', type=str, 
                        help='Comma-separated string of maximum-tree-depths to sweep over.')
    parser.add_argument('--n_estimators', type=str, default=100, 
                        help='Number of estimators for ensemble methods.')

    # Save and print settings
    parser.add_argument('--sweep_name', type=str, default="single_run",
                        help='Name of sweep.')
    parser.add_argument('--is_large_sweep', action='store_true',
                        help='Hide cross-val progress and all prints.')
    parser.add_argument('--results_dir', type=str, default="results/",
                        help='Path to save results.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print statistics for insight/debugging.')

    args_ = parser.parse_args()

    # Default settings (TODO: move these to a config file)
    if args_.weight_decays is None:
        args_.weight_decays = list(np.logspace(-4, 4, 9))
    else:
        args_.weight_decays = [float(wd) for wd in args_.weight_decays.split(",")]

    if args_.max_depths is None:
        args_.max_depths = [1, 2, 5, 10, 15, 20, None]
    else:
        args_.max_depths = [int(md) for md in args_.max_depths.split(",")]

    main(args_)
