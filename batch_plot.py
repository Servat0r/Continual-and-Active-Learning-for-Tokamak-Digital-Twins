from joblib import Parallel, delayed
import os
from rich import print

#simulator_types = ['qualikiz', 'tglf']
simulator_types = ['tglf']
pow_types = ['highpow', 'lowpow', 'mixed']
cluster_types = ['tau_based', 'Ip_Pin_based', 'wmhd_based', 'beta_based']

def generator():
    for simulator_type in simulator_types:
        for pow_type in pow_types:
            for cluster_type in cluster_types:
                if pow_type in ['highpow', 'lowpow'] and cluster_type in ['tau_based', 'Ip_Pin_based']:
                    yield simulator_type, pow_type, cluster_type
                elif pow_type == 'mixed' and cluster_type in ['wmhd_based', 'beta_based']:
                    yield simulator_type, pow_type, cluster_type


def task(simulator_type, pow_type, cluster_type):
    if simulator_type == 'tglf':
        split = 0 #9
    else:
        if pow_type == 'lowpow':
            split = 8
        else:
            split = 9
    os.system(
        f"python plot.py --simulator_type={simulator_type} --pow_type={pow_type} --cluster_type={cluster_type} --dataset_type=not_null --start=0 --end={split}"
    )
    os.system(
        f"python plot.py --simulator_type={simulator_type} --pow_type={pow_type} --cluster_type={cluster_type} --dataset_type=not_null --start={split}"
    )
    print(f"[red]Done with {simulator_type}-{pow_type}-{cluster_type}[/red]")


if __name__ == '__main__':
    Parallel(n_jobs=os.cpu_count())(
        delayed(task)(simulator_type, pow_type, cluster_type) for simulator_type, pow_type, cluster_type in generator()
    )
