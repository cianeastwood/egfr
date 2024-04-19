""" Main file for evaluating experiments. """

import argparse
import os
import glob
from functools import partial
from io import StringIO
import pandas as pd

from data.loader import EGFRDataset
from train import get_model_and_hparams
from utils import seed_everything, model_performance, plot_roc_curves

pd.set_option("display.precision", 2)


def load_results(results_dir):
    """ Load results from jsonl files to a pandas dataframe."""
    records = []
    for fname in glob.glob(os.path.join(results_dir, "*.jsonl")):
        with open(fname, "r") as f:
            if os.path.getsize(fname) != 0:
                records.append(f.readline().strip())

    df = pd.read_json(StringIO("\n".join(records)), lines=True)

    return df


def filter_df_dict_column(row, dict_column_name, filter_dict):
    """Filter a dataframe based on a dict-type column."""
    df_dict = dict(row[dict_column_name])
    for k, v in filter_dict.items():
        if k not in df_dict:
            raise ValueError(f"Key '{k}' not in df_dict with keys: {df_dict.keys()}")
        if df_dict[k] != v:
            return False
    return True


def main(args):
    """ Main function for evaluating results."""
    # Load results
    df = load_results(args.results_dir)

    # Optionally filter
    if args.arg_values is not None:
        filter_fn = partial(
            filter_df_dict_column, 
            dict_column_name="args", 
            filter_dict=args.arg_values
        )
        mask = df.apply(filter_fn, axis=1)
        df = df[mask]

    # Report test results for the best model of each type
    models = []
    results = []
    for m_name in df["model_name"].unique():
        seed_everything()

        # filtered by model
        df_m = df[df["model_name"] == m_name]

        # find the entry with the best score
        best_run = df_m["score"].idxmax()

        # extract info
        best_hparams = df_m.loc[best_run, "hparams"]
        m_args = df_m.loc[best_run, "args"]  # argparse.Namespace
        m_args = argparse.Namespace(**m_args)

        # Load dataset with these settings (best features)
        dataset = EGFRDataset(
            data_dir=m_args.data_dir,
            fp_methods=m_args.fp_methods,
            pretrained_features=m_args.pretr_feat,
        )
        X_train, y_train = dataset.X_train, dataset.y_train
        X_test, y_test = dataset.X_test, dataset.y_test
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        # Get model and set best hparams
        model, _ = get_model_and_hparams(m_args, n_pos, n_neg)
        model.set_params(**best_hparams)

        # Train on (full) train-val set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Report results
        m_results = {
            "model": m_name,
            "fingerpr": m_args.fp_methods,
            "pretr_feat": m_args.pretr_feat,
        }
        m_perf = model_performance(y_test, y_pred, y_prob, verbose=False)
        m_results.update(m_perf)
        if args.show_val_score:
            m_results["f2_val"] = df_m.loc[best_run, "score"]
        results.append(m_results)

        # Store models for plotting
        if args.plot:
            models.append({"label": m_name.upper(), "y_pred": y_pred, "y_prob": y_prob})

    # Print results using pandas dataframe
    results_df = pd.DataFrame(results).sort_values(m_args.eval_metric, ascending=False)
    if args.print_markdown:
        print(results_df.to_markdown(index=False, tablefmt="github", floatfmt=".2f"))
    else:
        print(results_df.to_string(index=False))

    if args.plot:
        plot_save_path = os.path.join(args.results_dir, "roc_curve.png")
        plot_roc_curves(models, y_test, save_path=plot_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/", help="Path to data.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/my_sweep",
        help="Path to sweep results.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate plots for comparing models."
    )
    parser.add_argument(
        "--show_val_score",
        action="store_true",
        help="Show validation-set score for comparison.",
    )
    parser.add_argument(
        "--print_markdown",
        action="store_true",
        help="Print results in markdown format.",
    )
    parser.add_argument(
        "--arg_values",
        type=str,
        default=None,
        help="Filter results by argument values.",
    )

    args_ = parser.parse_args()

    if args_.arg_values is not None:
        # pass in arguments and their values, e.g.: 'pretr_feat=chemberta,fp_methods=fcfp'
        kvs = [kv.split("=") for kv in args_.arg_values.split(",")]
        args_.arg_values = {
            kv[0]: kv[1] for kv in kvs
        }  # infer type of string, e.g. float, int

    main(args_)
