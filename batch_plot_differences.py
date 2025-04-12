import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = [
     """0.035 0.018 0.017 0.010 0.009 0.014 0.022 0.031 0.026
0.029 0.032 0.033 0.035 0.028 0.022 0.019 0.021 0.023
0.038 0.025 0.028 0.027 0.025 0.016 0.014 0.016 0.014
0.033 0.025 0.023 0.019 0.019 0.014 0.012 0.014 0.012
0.038 0.029 0.028 0.027 0.021 0.021 0.015 0.016 0.012""",
    """0.013 0.027 0.004 0.008 0.000 0.016 0.006 0.014 0.037
0.030 0.028 0.028 0.023 0.015 0.021 0.016 0.023 0.024
0.020 0.023 0.029 0.020 0.017 0.022 0.018 0.020 0.019
0.015 0.013 0.029 0.018 0.014 0.014 0.011 0.015 0.015
0.033 0.038 0.042 0.030 0.028 0.027 0.010 0.024 0.021""",
    """0.017 0.021 0.031 0.011 0.002 0.013 0.017 0.016 0.052
0.015 0.014 0.016 0.016 0.014 0.015 0.015 0.014 0.016
0.011 0.010 0.014 0.013 0.018 0.015 0.014 0.011 0.011
0.008 0.012 0.010 0.012 0.011 0.013 0.010 0.011 0.007
0.008 0.013 0.005 0.016 0.012 0.012 0.011 0.011 0.011""",
    """0.020 0.020 0.044 0.024 0.001 0.015 0.025 0.023 0.018
0.041 0.039 0.043 0.031 0.025 0.023 0.021 0.018 0.019
0.023 0.020 0.045 0.017 0.015 0.014 0.012 0.008 0.008
0.003 -0.007 0.000 -0.007 0.000 0.002 0.005 0.004 0.005
0.020 0.021 0.018 0.018 0.018 0.016 0.017 0.013 0.014""",
    """0.016 0.019 0.036 0.026 0.029 0.035 0.023 0.036 0.032
0.046 0.049 0.048 0.046 0.044 0.045 0.040 0.038 0.034
0.029 0.053 0.048 0.042 0.042 0.041 0.037 0.031 0.026
0.025 0.036 0.035 0.035 0.032 0.034 0.026 0.024 0.021
0.044 0.041 0.038 0.045 0.042 0.041 0.033 0.040 0.033""",
    """0.037 0.042 0.024 0.003 0.013 0.017 0.008 0.011 0.036
0.037 0.043 0.030 0.026 0.023 0.024 0.022 0.023 0.022
0.040 0.047 0.039 0.038 0.026 0.020 0.019 0.019 0.019
0.036 0.025 0.027 0.027 0.016 0.017 0.017 0.016 0.016
0.030 0.032 0.024 0.018 0.015 0.016 0.018 0.017 0.014"""
]


pow_types = ['highpow', 'lowpow', 'mixed']
cluster_types = ['tau_based', 'Ip_Pin_based', 'wmhd_based', 'beta_based']


def get_gem(pow_type, cluster_type):
     if pow_type == 'lowpow':
          return 200
     else:
          return 400


def get_df(string: str, savepath: str, gem: int = 400, show: bool = False):
	lines = string.strip().split('\n')
	assert len(lines) == 5
	finals = []
	for line in lines:
		finals.append([float(item) for item in line.split(' ')])
	df = pd.DataFrame({
		'Experience': list(range(1, 10)),
		'Naive': finals[0],
		'Cumulative': finals[4],
		'Replay (2000)': finals[1],
		f'GEM ({gem})': finals[2],
		'GEM (1024)': finals[3]
	})
	print(df)
	df.plot(x='Experience', kind='line')
	plt.grid(True)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=12)
	plt.savefig(savepath)
	if show: plt.show()


def is_valid(pow_type, cluster_type):
    if pow_type == 'mixed':
        return cluster_type in cluster_types[2:]
    else:
        return cluster_type in cluster_types[:2]


if __name__ == '__main__':
    index = -1
    for pow_type in pow_types:
        for cluster_type in cluster_types:
            if is_valid(pow_type, cluster_type):
                index += 1
                gem = get_gem(pow_type, cluster_type)
                string = data[index]
                savepath = f"plots/AL(CL)/lcmd_random_difference_qlk_{pow_type}_{cluster_type[:-6]}_256B_1024M.png"
                get_df(string, savepath, gem, show=True)
