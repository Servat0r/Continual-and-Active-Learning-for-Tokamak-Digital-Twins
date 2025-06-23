import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.ex_post_tests import load_complete_dataset
from plot_averages.common import colors
from argparse import ArgumentParser


def get_y_x_sizes(total: int):
    y_size, x_size = 1, total
    for candidate_y in range(int(math.sqrt(total)), 0, -1):
        if total % candidate_y == 0:
            y_size, x_size = candidate_y, total // candidate_y
            break
    return y_size, x_size


def get_direction_name(cluster_type: str):
    data = {
        'tau_based': r"$\tau = \dfrac{W_{mhd}}{P_{in}}$",
        'Ip_Pin_based': r"$I_p \times P_{in}$",
        'wmhd_based': r"$W_{mhd}$",
        'beta_based': r"$\beta$" 
    }
    return data[cluster_type]


def get_dict_keys(simulator_type: str):
    if simulator_type == 'qualikiz':
        dict_keys = {
            "ti_te0": r"$T_i/Te$", 'Zeff': r'$Z_{eff}$', 'Ati0': r'$R/L_{T_i}$', #'ZD_FZEFF': 'Zeff', 'Ati0': r'$R/L_{T_i}$',
            'Ate': r'$R/L_{T_e}$', 'Ane': r'$R/L_{n_e}$', 'x': r'$r/a$',
            'q': '$q$', 'smag': r'$\hat{s}$', 'alpha': r'$\alpha$',
            'Ani1': r'$R/L_{T_{imp}}$', 'normni1': r'$n_{imp,light}/n_e$',
            'logNustar': r'$\log{\nu^*}$', 'autor': r'$R/L_{tor}$',
            'machtor': r'M_{tor}', 'gammae': r'$\gamma_E$', 'efe': r'$q_e$',
            'efi': r'$q_i$', 'pfi': r'$\Gamma_i$', 'pfe': r'$\Gamma_e$'
        }
    elif simulator_type == 'tglf':
        dict_keys = {
            "taus_2": r"$T_i/Te$", 'zeff': 'Zeff', 'kappa_loc': r'$\kappa$',
            'drmajdx_loc': r'$\dfrac{\partial R_{maj}}{\partial x}$',
            'rlts_2': r'$R/L_{T_i}$', 'rlts_1': r'$R/L_{T_e}$', 'rlns_1': r'$R/L_{n_e}$',
            'betae': r'$\beta_e$' , 'rmin_loc': r'$r/a$', 'q_loc': r'$q$',
            'delta_loc': r'$\delta$', 'vexb_shear': r'$E \times B$ shear',
            'q_prime_loc': r'$\approx \hat{s}$', 'rlns_3': r'$R/L_{T_{imp}}$',
            'as_2': r'$n_i/n_e$', 'xnue': r'$\nu_{ee}$', 'efe': r'$q_e$',
            'efi': r'$q_i$', 'pfi': r'$\Gamma_i$', 'pfe': r'$\Gamma_e$'
        }
    else:
        raise ValueError(f"Unknown or not implemented simulator_type = \"{simulator_type}\"")
    dict_keys = {key.lower(): value for key, value in dict_keys.items()}
    return dict_keys


def plot(
    df: pd.DataFrame, filesave: str, direction_name: str, dict_keys: dict,
    vars_of_interest: list[str], xbins: list, show: bool = True, simulator_type: str = 'qualikiz'
):
    MIN = np.min(df['direction'])
    MAX = np.max(df['direction'])

    ybins = np.linspace(MIN, MAX, 11)
    # Try to make the final plots figure as "squarey" as possible
    total_length = len(vars_of_interest)
    y_size, x_size = get_y_x_sizes(total_length)
    #y_size2, x_size2 = get_y_x_sizes(total_length + 1)
    #r1, r2 = x_size / y_size, x_size2 / y_size2
    #if r1 > r2:
    #    y_size, x_size = y_size2, x_size2
    print(f"y_size = {y_size}, x_size = {x_size}")
    fig, ax = plt.subplots(y_size, x_size, figsize=(16,7))#, gridspec_kw={'right': 0.8})

    print("Before cycling for plots ...")
    for inp,this_ax in zip(vars_of_interest, ax.ravel()):
        for j in range(len(xbins)-1):
            #"""
            means_ybins = []
            std_ybins = []
            for i in range(len(ybins)):
                if simulator_type == 'qualikiz':
                    tmp = df.query(f"campaign=={i} & x>{xbins[j]} & x<{xbins[j+1]}")
                else:
                    tmp = df.query(f"campaign=={i} & rmin_loc>{xbins[j]} & rmin_loc<{xbins[j+1]}")
                means_ybins.append(tmp[inp].mean())
                std_ybins.append(tmp[inp].std())
            """
            means_ybins = {}
            std_ybins = {}
            available = []
            for i in range(len(ybins)):
                if simulator_type == 'qualikiz':
                    tmp = df.query(f"campaign=={i} & x>{xbins[j]} & x<{xbins[j+1]}")
                else:
                    tmp = df.query(f"campaign=={i} & rmin_loc>{xbins[j]} & rmin_loc<{xbins[j+1]}")
                if len(tmp) > 0:
                    means_ybins[i] = tmp[inp].mean()
                    std_ybins[i] = tmp[inp].std()
                    available.append(i)
            if len(available) > 0:
                for idx, i in enumerate(available[:-1]):
                    j2 = available[idx+1]
                    if j2 > i+1:
                        mean_diff = (means_ybins[j2] - means_ybins[i]) / (j2 - i)
                        for k in range(i+1, j2):
                            means_ybins[k] = means_ybins[i] + mean_diff * (k - i)
                        std_diff = (std_ybins[j2] - std_ybins[i]) / (j2 - i)
                        for k in range(i+1, j2):
                            std_ybins[k] = std_ybins[i] + std_diff * (k - i)
                tmp_means_ybins = []
                tmp_std_ybins = []
                for idx in range(len(ybins)):
                    mean_val, std_val = means_ybins.get(idx), std_ybins.get(idx)
                    if mean_val is None:
                        if idx > 0:
                            mean_val = means_ybins[idx - 1]
                        else:
                            mean_val = means_ybins[idx + 1]
                    if std_val is None:
                        if idx > 0:
                            std_val = std_ybins[idx - 1]
                        else:
                            std_val = std_ybins[idx + 1]
                    tmp_means_ybins.append(mean_val)
                    tmp_std_ybins.append(std_val)
                means_ybins, std_ybins = tmp_means_ybins, tmp_std_ybins
                #means_ybins = [means_ybins[idx] if idx in means_ybins else 0.0 for idx in range(len(ybins))]
                #std_ybins = [std_ybins[idx] if idx in std_ybins else 0.0 for idx in range(len(ybins))]
            """
                #this_ax.plot(range(len(shotbins)), means_shots,  lw=2, label='shotbins')
            # means_ybins = np.abs(means_ybins)
            this_ax.plot(ybins, means_ybins,  lw=2, color=colors[j], label=rf'$r/a$={np.round((xbins[j]+xbins[j+1])/2,2)}',)
            # this_ax.fill_between(ybins, np.array(means_ybins)-np.array(std_ybins), np.array(means_ybins)+np.array(std_ybins), color=colors[j], alpha=0.1)

            this_ax.set_ylabel(dict_keys[inp] + f" ({inp})")
            this_ax.set_xlabel(direction_name)        #this_ax.set_yscale('log')
            this_ax.grid(True)

        #this_ax.set_ylim(0.5,250)
    #ax[0][0].legend(ncol=2)
    #ax[0][0].legend(ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    if y_size > 1:
        ax[-1][-1].legend(loc='center right', bbox_to_anchor=(1.40, 0.5))#(1.70, 0.5))
    else:
        ax[-1].legend(loc='center right', bbox_to_anchor=(1.40, 0.5))

    fig.tight_layout()
    if not os.path.exists(os.path.dirname(filesave)):
        os.makedirs(os.path.dirname(filesave))
    plt.savefig(filesave)
    if show:
        plt.show()
    #ax[0][0].legend()


def mainloop(
    simulator_type: str, pow_type: str, cluster_type: str, dataset_type: str,
    inputs_for_plot: list[str], show: bool = True, start: int = 0, end: int = 0
):
    df = load_complete_dataset(pow_type, cluster_type, dataset_type, simulator_type)
    xbins = np.linspace(0.0, 1.0, 11)
    direction_name = get_direction_name(cluster_type)
    dict_keys = get_dict_keys(simulator_type)
    filesave = f'plots/Distribution Shifts_{simulator_type}_{pow_type}_{cluster_type}_{dataset_type}_{start}_to_{end}.png'
    for column in inputs_for_plot:
        series = df[column]
        print(f"{column} => min = {series.min():.4f}, max = {series.max():.4f}, mean = {series.mean():.4f}, std = {series.std():.4f}")
    plot(df, filesave, direction_name, dict_keys, inputs_for_plot, xbins, show=show, simulator_type=simulator_type)


def qlk_loop(
    pow_type: str, cluster_type: str, dataset_type: str, start: int = 0, end: int = 0, show=False
):
    simulator_type = 'qualikiz'
    if pow_type in ['highpow', 'mixed']:
        inputs_for_plots = [ # Excluded 'autor', 'machtor', 'gammae', 'x' (length = 11)
            'ane', 'ate', 'zeff', 'q', 'smag', 'alpha',
            'ani1', 'ati0', 'normni1', 'ti_te0', 'lognustar'
        ]
    else:
        inputs_for_plots = [
            'ane', 'ate', 'q', 'smag', 'alpha', 'ani1',
            'ati0', 'normni1', 'zeff', 'lognustar' # Excluded 'x' (length = 10)
        ]
    inputs_for_plots.extend(['efe', 'efi', 'pfe', 'pfi'])
    if end <= 0:
        end += len(inputs_for_plots)
    inputs_for_plots = inputs_for_plots[start:end]
    mainloop(
        simulator_type, pow_type, cluster_type, dataset_type,
        inputs_for_plots, start=start, end=end, show=show
    )


def tglf_loop(
    pow_type: str, cluster_type: str, dataset_type: str, start: int = 0, end: int = 0, show=False
):
    simulator_type = 'tglf'
    inputs_for_plots = [ # Excluded "rmin_loc", "rlns_2", "vexb_shear" (length = 14)
        "drmajdx_loc", "kappa_loc", "delta_loc", "q_loc",
        "q_prime_loc", "rlns_1", "rlts_1", "rlts_2",
        "taus_2", "as_2", "rlns_3", "zeff", "betae", "xnue"
    ]
    inputs_for_plots.extend(['efe', 'efi', 'pfe', 'pfi'])
    if end <= 0:
        end += len(inputs_for_plots)
    inputs_for_plots = inputs_for_plots[start:end]
    mainloop(
        simulator_type, pow_type, cluster_type, dataset_type,
        inputs_for_plots, start=start, end=end, show=show
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simulator_type', type=str, default='qualikiz')
    parser.add_argument('--pow_type', type=str, default='highpow')
    parser.add_argument('--cluster_type', type=str, default='tau_based')
    parser.add_argument('--dataset_type', type=str, default='not_null')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()
    if args.simulator_type == 'qualikiz':
        qlk_loop(args.pow_type, args.cluster_type, args.dataset_type, start=args.start, end=args.end, show=args.show)
    elif args.simulator_type == 'tglf':
        tglf_loop(args.pow_type, args.cluster_type, args.dataset_type, start=args.start, end=args.end, show=args.show)
