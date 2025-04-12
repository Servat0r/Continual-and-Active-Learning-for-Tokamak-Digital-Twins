import os

from src.utils import *
from src.ex_post_tests import *
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--strategy', type=str, default='Naive')
    parser.add_argument('--extra_log_folder', type=str, default='Base')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--al_batch_size', type=int, default=256)
    parser.add_argument('--al_max_batch_size', type=int, default=1024)
    #parser.add_argument('--full_first_set', type=bool, default=False)
    #parser.add_argument('--reload_weights', type=bool, default=False)
    parser.add_argument('--downsampling', type=float, default=0.5)
    parser.add_argument('--ncampaigns', type=int, default=10)
    parser.add_argument('--index', type=int, default=-1)

    args = parser.parse_args()

    for al_method in ['random_sketch_grad', 'batchbald', 'badge', 'lcmd_sketch_grad']:
        system_call = f"python latex_compute.py --mode=al_cl --simulator_type={args.simulator_type} " + \
        f"--pow_type={args.pow_type} --cluster_type={args.cluster_type} --strategy={args.strategy} " + \
        f"--extra_log_folder=\"{args.extra_log_folder}\" --hidden_size={args.hidden_size} --al_method={al_method} " + \
        f"--al_batch_size={args.al_batch_size} --al_max_batch_size={args.al_max_batch_size} --full_first_set=1 " + \
        f"--index={args.index} --time_type=total"
        os.system(system_call)
