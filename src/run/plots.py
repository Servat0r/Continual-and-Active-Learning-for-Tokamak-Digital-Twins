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


def mean_std_evaluation_experiences_plots(
        file_paths, metric_list, title_list, ylabel_list, start_exp=0, end_exp=9, num_exp=None, set_type='eval'
):
    save_folder = os.path.dirname(file_paths[0])
    for metric, title, ylabel in zip(metric_list, title_list, ylabel_list):
        if start_exp == 0 and end_exp >= 4:
            # Experiences 0-4
            plot_metric_over_evaluation_experiences_multiple_runs(
                file_paths, metric, title, 'Training Experience',
                ylabel, show=False, experiences=range(5), num_exp=num_exp,
                savepath=os.path.join(save_folder, f'{set_type}_mean_std_plot_of_first_5_experiences_{metric[:-4]}.png'),
                from_beginning=False,
            )
        # Plot over all experiences
        plot_metric_over_evaluation_experiences_multiple_runs(
            file_paths, metric, title, 'Training Experience', ylabel,
            show=False, experiences=range(start_exp, end_exp+1), num_exp=num_exp,
            savepath=os.path.join(save_folder, f'{set_type}_mean_std_plot_of_all_10_experiences_{metric[:-4]}.png'),
            from_beginning=False,
        )


__all__ = [
    'evaluation_experiences_plots',
    'mean_std_evaluation_experiences_plots',
]