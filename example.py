import numpy as np
import matplotlib.pyplot as plt
from src.utils.scenarios import *
from src.utils.logging import *
from src.ex_post_tests import *


plt.rcParams['figure.figsize'] = (8, 6)

clusters = ['wmhd', 'beta']
strategy_names = ['Naive', 'Replay', 'Replay', 'Cumulative']
extra_log_folders = ['Base', 'Buffer 2000', 'Buffer 10000', 'Base']
strategy_tags = ['naive', 'replay2000', 'replay10000', 'cumulative']
#strategy_names = ['Cumulative']
#extra_log_folders = ['Base']
#strategy_tags = ['cumulative']


for cluster in clusters:
    scenario = ScenarioConfig('qualikiz', 'mixed', f'{cluster}_based', 'not_null', 'regression')
    for strategy_name, extra_log_folder, strategy_tag in zip(strategy_names, extra_log_folders, strategy_tags):
        w = extract_dataset_weights(scenario, 'final', weights_source='test')
        print(w)
        config = LoggingConfiguration(
            scenario, strategy=strategy_name, extra_log_folder=extra_log_folder,
            hidden_size=1024, hidden_layers=2, batch_size=4096
        )
        if cluster == 'wmhd':
            title_str = r"$R^2$ - $W_{MHD}$-based Scenario"
        elif cluster == 'beta':
            title_str = r"$R^2$ - $\beta_N$-based Scenario"
        elif cluster == 'tau':
            title_str = r"$R^2$ - $\tau$-based Scenario"
        elif cluster == 'Ip_Pin':
            title_str = r"$R^2$ - $I_p \times P_{in}$-based Scenario"
        else:
            raise ValueError(f"Unknown cluster {cluster}")
        #mean_and_separated_plots(
        #    config, internal_metric_name='R2Score_Exp', plot_metric_name=r"$R^2$", count=-1, save=True,
        #    savepath=f'plots/arithmetic_mean_and_separated_r2score_qlk_mixed_{cluster}_{strategy_tag}.pdf',
        #    weighted_means=False, show=True, include_std=False, title=title_str
        #)
        if strategy_name == 'Cumulative':
            y_values = np.arange(start=0.84, stop=0.94, step=0.02)
        elif strategy_name == 'Naive':
            if cluster == 'wmhd':
                y_values = np.arange(start=0.76, stop=0.94, step=0.02)
            else:
                y_values = np.arange(start=0.66, stop=0.94, step=0.02)
        else:
            y_values = np.arange(start=0.78, stop=0.94, step=0.02)
        mean_and_separated_plots(
            config, internal_metric_name='R2Score_Exp', plot_metric_name=r"$R^2$", count=-1, save=True,
            savepath=f'plots/weighted_mean_and_separated_r2score_qlk_mixed_{cluster}_{strategy_tag}.pdf',
            weighted_means=True, show=True, include_std=False, title=title_str,
            title_size=20, xlabel_size=16, ylabel_size=16,
            x_values=np.arange(start=1, stop=11, step=1), y_values=y_values
        )
