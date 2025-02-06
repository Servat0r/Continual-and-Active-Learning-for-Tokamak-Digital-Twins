import argparse


def build_argparser():
    # Argument Parsing
    cmd_arg_parser = argparse.ArgumentParser(
        description="Parse command-line options with specific parameters."
    )
    cmd_arg_parser.add_argument(
        '--config',
        type=str,
        help="JSON configuration file to load."
    )
    cmd_arg_parser.add_argument(
        '--num_tasks',
        type=int,
        help="Number of tasks to execute (e.g., --num_tasks=16). If <= 0, it defaults to os.cpu_count() // 2.",
    )
    cmd_arg_parser.add_argument(
        '--tasks',
        nargs='+', type=int, default=None,
        help='Specific tasks for filtering which ones to execute.'
    )
    cmd_arg_parser.add_argument(
        '--extra-log-folder',
        type=str,
        help="Extra folder to be added to log_folder path, if necessary."
    )
    cmd_arg_parser.add_argument(
        '--no-redirect-stdout',
        action='store_true',
        help='Disable standard output redirection to a file. Defaults to False (i.e., output is redirected).',
    )
    cmd_arg_parser.add_argument(
        '--write-intermediate-models',
        action='store_true',
        help='Allows to save checkpoints of intermediate models after each experience.',
    )
    cmd_arg_parser.add_argument(
        '--plot-single-runs',
        action='store_true',
        help='Allows to plot graphs for each single run.',
    )
    cmd_arg_parser.add_argument(
        '--test',
        action='store_true',
    )
    return cmd_arg_parser
