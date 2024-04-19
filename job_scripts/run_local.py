""" Run jobs/commands on local machine. """

import argparse
import subprocess
import tqdm


def local_launcher(cmds):
    """Launch commands serially on the local machine."""
    with tqdm.tqdm(total=len(cmds)) as progress_bar:
        for cmd in cmds:
            subprocess.call(cmd, shell=True)
            progress_bar.update(1)


if __name__ == '__main__':
    # Flags
    parser = argparse.ArgumentParser(description='Run jobs on local machine.')
    parser.add_argument('--commands_fpath', type=str, help="File with one command per line.")
    args = parser.parse_args()

    # Load commands from rgs.commands_file file with one command per line
    f = open(args.commands_fpath, "r")
    commands = f.read().split("\n")
    print(f'Running {len(commands)} jobs...')

    # Run jobs
    local_launcher(commands)
    print("All jobs completed.")
