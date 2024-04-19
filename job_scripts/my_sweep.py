""" Generate commands for a sweep. """

import argparse
import os

def build_commands(args):
    """ Build commands for the sweep. """
    # Things to sweep over
    pretrain_features = [None, "fcd", "chemberta"]
    fingerprints = [None, "fcfp", "ecfp", "maccs", "maccs,fcfp", "maccs,ecfp"]
    model_names = ["linear", "rf", "gbt"]

    # Base command
    base_cmd = f"python train.py --sweep_name {args.sweep_name} --is_large_sweep"
    if args.data_dir is not None:
        base_cmd += f" --data_dir {args.data_dir}"
    if args.results_dir is not None:
        base_cmd += f" --results_dir {args.results_dir}"

    # Generate full commands
    os.makedirs(args.jobs_dir, exist_ok=True)
    output_fpath = os.path.join(args.jobs_dir, f"{args.sweep_name}.txt")
    output_file = open(output_fpath, "w")
    for pretr_feat in pretrain_features:
        pretr_cmd = base_cmd
        if pretr_feat is not None:
            pretr_cmd += f" --pretr_feat {pretr_feat}"
        for fp_methods in fingerprints:
            fp_cmd = pretr_cmd
            if pretr_feat is None and fp_methods is None:
                continue
            if fp_methods is not None:
                fp_cmd += f" --fp_methods {fp_methods}"
            for model_name in model_names:
                m_cmd = fp_cmd + f" --model_name {model_name}"
                is_last_line = (pretr_feat == pretrain_features[-1] and
                                fp_methods == fingerprints[-1] and
                                model_name == model_names[-1])
                if is_last_line:
                    print(m_cmd.strip(), file=output_file, end="")
                else:
                    print(m_cmd.strip(), file=output_file)

    output_file.close()
    output_file = open(output_fpath , "r")
    print(f'Total num experiments = {len(output_file.readlines())}')


if __name__ == '__main__':
    # Flags
    parser = argparse.ArgumentParser(description='Generate commands for experiments.')
    parser.add_argument('--data_dir', type=str, help="Absolute path to data directory.")
    parser.add_argument('--results_dir', type=str, help="Absolute path to results directory.")
    parser.add_argument('--jobs_dir', type=str, default="job_scripts/jobs/", help="Dir for jobs.")
    parser.add_argument('--sweep_name', type=str, default="my_sweep", help="Name of the sweep.")
    args_ = parser.parse_args()

    build_commands(args_)
