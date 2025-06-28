# Computation of derived metrics for Experiments
from typing import Literal, Optional, Any
import numpy as np
from ..database import SecureMLExperimentDB, Experiment, Strategy

# Get experiment dict and logging config
# Build Naive experiment dict and retrieve Naive experiment, or give an error
# Build Cumulative experiment dict and retrieve Cumulative experiment, or give an error
# Compute metrics as in dbCompute.py
# Save metrics to database

__exp_ids = {
    'id_general', 'id_scenario', 'id_architecture', 'id_loss', 'id_optimizer',
    'id_scheduler', 'id_early_stopping', 'id_strategy', 'id_active_learning'
}


def compute_derived_metrics(
    db: SecureMLExperimentDB, exp_id: int, exp_name: str, set_type: Optional[Literal['eval', 'test']],
    which: Literal['R', 'time_ratios', 'all'] = 'all', exp_dict: dict[str, Any] = None
) -> dict:
    # Which metrics to compute
    match which:
        case 'R':
            computeR, computeT = True, False
        case 'time_ratios':
            computeR, computeT = False, True
        case 'all':
            computeR = computeT = True
        case _:
            raise ValueError(f"Unknown value of \"which\": {which}")
    print()
    if not exp_dict:
        exp_dict = db.get_one_by_id(Experiment, exp_id)
    new_log_dict = {}
    # Check names are correct
    assert exp_dict['name'] == exp_name, \
        f"Experiment name in database is different from passed parameter: {exp_dict['name']} vs {exp_name}"

    # Retrive ids of Naive and Cumulative strategies
    naive_query_dict, cumulative_query_dict = {'name': 'Naive'}, {'name': 'Cumulative'}
    naive_data = db.get_first(Strategy, naive_query_dict)
    print(naive_data)
    naive_id = naive_data['id']
    # Build conditions for retrieving Naive and Cumulative experiments
    naive_query_dict = {k: v for k, v in exp_dict.items() if k in __exp_ids}
    naive_query_dict['id_strategy'] = naive_id
    naive_query_dict['status'] = 'finished'
    naive_query_dict['is_test'] = False
    #stdout_debug_print(f"Naive query dict: {naive_query_dict}", color='red')
    naive_exp_dict = db.get(Experiment, naive_query_dict)[-1] # By default, last experiment in list
    naive_log_dict = naive_exp_dict['logs']['aggregated_metrics']

    #stdout_debug_print(f"Strategy exp dict: {exp_dict}", color='red')
    strategy_log_dict = exp_dict['logs']['aggregated_metrics']

    if computeR:
        # Retrieve cumulative data
        cumulative_data = db.get_first(Strategy, cumulative_query_dict)
        cumulative_id = cumulative_data['id']
        ## Cumulative
        cumulative_query_dict = {k: v for k, v in exp_dict.items() if k in __exp_ids}
        cumulative_query_dict['id_strategy'] = cumulative_id
        cumulative_query_dict['status'] = 'finished'
        cumulative_query_dict['is_test'] = False
        # Now retrieve Naive and Cumulative exp dicts
        #stdout_debug_print(f"Cumulative query dict: {cumulative_query_dict}", color='red')
        cumulative_exp_dict = db.get(Experiment, cumulative_query_dict)[-1] # By default, last experiment in list
        cumulative_log_dict = cumulative_exp_dict['logs']['aggregated_metrics']
        # Now compute "R" from "R2"
        naive_r2_values = np.array(naive_log_dict[f"{set_type.capitalize()}_R2"]['mean'])
        cumulative_r2_values = np.array(cumulative_log_dict[f"{set_type.capitalize()}_R2"]['mean'])
        strategy_r2_values = np.array(strategy_log_dict[f"{set_type.capitalize()}_R2"]['mean'])
        r_values = ((strategy_r2_values - naive_r2_values) / (cumulative_r2_values - naive_r2_values)).round(4)
        new_log_dict[f"{set_type.capitalize()}_R"] = r_values.tolist()
    if computeT:
        # Compute "time_ratios" from "times"
        naive_cumulative_times = np.array(naive_log_dict["cumulative_times"])
        strategy_cumulative_times = np.array(strategy_log_dict["cumulative_times"])
        time_ratios_values = (strategy_cumulative_times / naive_cumulative_times).round(4)
        new_log_dict['time_ratios'] = time_ratios_values.tolist()
    return new_log_dict


__all__ = ['compute_derived_metrics']
