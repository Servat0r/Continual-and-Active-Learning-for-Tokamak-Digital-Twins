import os

from src.utils import *
from src.ex_post_tests import *
from rich import print


simulator_type = 'tglf'
pow_type = 'lowpow'
cluster_type = 'Ip_Pin_based'
hidden_size = 256 #512
index = -2

_strategies = {
    'Naive': ['Base'],
    'FromScratchTraining': ['Base'],
    'Cumulative': ['Base'],
    'Replay': ['Buffer 500', 'Buffer 2500'],
    #'Replay': ['Buffer 600', 'Buffer 3000'],
    'PercentageReplay': ['Percentage 5% Min 500', 'Percentage 10% Min 2500'],
    #'PercentageReplay': ['Percentage 5% Min 600', 'Percentage 10% Min 3000'],
    'EWC': ['Lambda 1', 'Lambda 10'],
    'EWCReplay': ['Lambda 1 Buffer 2000', 'Lambda 10 Buffer 2000'],
    #'EWCReplay': ['Lambda 1 Buffer 3000', 'Lambda 10 Buffer 3000'],
    'MAS': ['Lambda 1 Alpha 0.0'],
    'MASReplay': ['Lambda 1 Alpha 0.0 Buffer 2000'],
    #'MASReplay': ['Lambda 1 Alpha 0.0 Buffer 3000'],
    'GEM': ['Patterns 100', 'Patterns 400'],
    #'GEM': ['Patterns 100', 'Patterns 300'],
    'GEMReplay': ['Patterns 400 Buffer 2000'],
    #'GEMReplay': ['Patterns 300 Buffer 3000'],
    'LFL': ['Lambda 1'],
    'SI': ['Lambda 1', 'Lambda 0.1']
}

strategies = {
    'Naive': ['Base (drop = 0.25) (tris)'],
    'FromScratchTraining': ['Base (drop = 0.25) (tris)'],
    'Cumulative': ['Base (drop = 0.25) (tris)'],
    'Replay': ['Buffer 500 (drop = 0.25) (tris)', 'Buffer 2500 (drop = 0.25) (tris)', 'Buffer 5000 (drop = 0.25) (tris)'],
    #'Replay': ['Buffer 600', 'Buffer 3000'],
    'PercentageReplay': [
        'Percentage 5% Min 500 (drop = 0.25) (tris)',
        'Percentage 10% Min 2500 (drop = 0.25) (tris)'
    ],
    #'PercentageReplay': ['Percentage 5% Min 600', 'Percentage 10% Min 3000'],
    'EWC': ['Lambda 1 (drop = 0.25) (tris)', 'Lambda 0.1 (drop = 0.25) (tris)'],
    'EWCReplay': ['Lambda 1 Buffer 2000 (drop = 0.25) (tris)', 'Lambda 0.1 Buffer 2000 (drop = 0.25) (tris)'],
    #'EWCReplay': ['Lambda 1 Buffer 3000', 'Lambda 10 Buffer 3000'],
    'MAS': ['Lambda 1 Alpha 0.0 (drop = 0.25) (tris)'],
    'MASReplay': ['Lambda 1 Alpha 0.0 Buffer 2000 (drop = 0.25) (tris)'],
    #'MASReplay': ['Lambda 1 Alpha 0.0 Buffer 3000'],
    'GEM': ['Patterns 400 (drop = 0.25) (tris)'],
    #'GEM': ['Patterns 100', 'Patterns 300'],
    'GEMReplay': ['Patterns 400 Buffer 2000 (drop = 0.25) (tris)'],
    #'GEMReplay': ['Patterns 300 Buffer 3000'],
    'LFL': ['Lambda 1 (drop = 0.25) (tris)', 'Lambda 0.1 (drop = 0.25) (tris)'],
    'SI': ['Lambda 1 (drop = 0.25) (tris)', 'Lambda 0.1 (drop = 0.25) (tris)']
}

__strategies = {
    'Naive': ['Base (drop = 0.35)'],
    'Cumulative': ['Base (drop = 0.35)'],
    'Replay': ['Buffer 500 (drop = 0.35)'] #, 'Buffer 2500']
}


if __name__ == '__main__':
    for strategy, folders in strategies.items():
        for extra_log_folder in folders:
            system_call = f"python latex_compute.py --set_type=test --simulator_type={simulator_type} " + \
            f"--pow_type={pow_type} --cluster_type={cluster_type} --strategy={strategy} " + \
            f"--extra_log_folder=\"{extra_log_folder}\" --hidden_size={hidden_size} --index={index}"
            print(f"[red]{simulator_type}-{pow_type}-{cluster_type}-{strategy}-{extra_log_folder}: [/red]", end='')
            os.system(system_call)
