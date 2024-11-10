import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_over_evaluation_experiences(file_path_or_buf: str | pd.DataFrame, metric):
    df = pd.read_csv(file_path_or_buf) if isinstance(file_path_or_buf, str) else file_path_or_buf
    num_exp = len(df['eval_exp'].unique())
    print(num_exp)
    data = [[] for _ in range(num_exp)]
    for training_exp in range(num_exp):
        for eval_exp in range(num_exp):
            value = df[(df['training_exp'] == training_exp) & (df['eval_exp'] == eval_exp)][metric].iloc[0]
            data[eval_exp].append(value)
    dict_data = {}
    for i in range(num_exp):
        dict_data[f"Eval Experience {i}"] = np.array(data[i])
    print(dict_data)
    # Extract keys and values
    x_values = list(dict_data.keys())
    y_values = list(dict_data.values())

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o')  # Use 'o' for markers
    plt.xlabel("Keys")
    plt.ylabel("Values")
    plt.title("Plot of Dictionary Data")
    plt.grid(True)
    plt.show()


__all__ = ['plot_metric_over_evaluation_experiences']
