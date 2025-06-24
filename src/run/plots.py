import os

from ..utils import *


def evaluation_experiences_plots(log_folder, metric_list, title_list, ylabel_list):
    for metric, title, ylabel in zip(metric_list, title_list, ylabel_list):
        # Experiences 0-4
        plot_metric_over_evaluation_experiences(
            os.path.join(log_folder, 'eval_results_experience.csv'), metric,
            title, 'Training Experience', ylabel, show=False, experiences=range(5),
            savepath=os.path.join(log_folder, f'plot_of_first_5_experiences_{metric[:-4]}.png'),
            from_beginning=False,
        )
        # Plot over all experiences
        plot_metric_over_evaluation_experiences(
            os.path.join(log_folder, 'eval_results_experience.csv'), metric,
            title, 'Training Experience', ylabel, show=False, experiences=range(10),
            savepath=os.path.join(log_folder, f'plot_of_all_10_experiences_{metric[:-4]}.png'),
            from_beginning=False,
        )


__all__ = ['evaluation_experiences_plots']