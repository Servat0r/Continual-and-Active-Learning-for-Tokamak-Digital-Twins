import matplotlib.pyplot as plt
from src.ex_post_tests import *
from src.utils.logging import *


plt.rcParams['figure.figsize'] = (8, 6)

clusters = ['wmhd', 'beta']
#strategy_names = ['Naive', 'Replay', 'PercentageReplay', 'Cumulative']
#extra_log_folders = ['Base', 'Buffer 2000', 'Percentage 5% Min 2000', 'Base']
#strategy_tags = ['naive', 'replay2000', 'percentagereplay-0.05-2000', 'cumulative']
strategy_names = ['Replay']
extra_log_folders = ['Buffer 10000']
strategy_tags = ['replay10000']


for cluster in clusters:
    for strategy_name, extra_log_folder, strategy_tag in zip(strategy_names, extra_log_folders, strategy_tags):
        w = extract_dataset_weights(
            'qualikiz', 'mixed', f'{cluster}_based', 'not_null', 'final', 'regression', weights_source='test'
        )
        print(w)
        config = LoggingConfiguration(
            simulator_type='qualikiz', pow_type='mixed', cluster_type=f'{cluster}_based', dataset_type='not_null',
            task='regression', strategy=strategy_name, extra_log_folder=extra_log_folder
        )
        mean_and_separated_plots(
            config, internal_metric_name='R2Score_Exp', plot_metric_name='R2Score', count=-1, save=True,
            savepath=f'plots/arithmetic_mean_and_separated_r2score_qlk_mixed_{cluster}_{strategy_tag}.pdf',
            weighted_means=False, show=True, include_std=False
        )
        mean_and_separated_plots(
            config, internal_metric_name='R2Score_Exp', plot_metric_name='R2Score', count=-1, save=True,
            savepath=f'plots/weighted_mean_and_separated_r2score_qlk_mixed_{cluster}_{strategy_tag}.pdf',
            weighted_means=True, show=True, include_std=False
        )
