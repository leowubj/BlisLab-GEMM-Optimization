import matplotlib.pyplot as plt
import numpy as np

import os
# Set the working directory
os.chdir('/home/ubuntu/Documents/pa1-haw006-jiw015')

"""Data Region {"""
# Convert to strings to make them equally spaced
Ns = np.array([32, 64, 128, 256, 511, 512, 513, 1023, 1024, 1025, 2047, 2048]).astype(str)
Data_naive = np.array([3.6125, 3.208, 1.2386, 1.2297, 1.6595, 1.09285, 1.6675, 1.645, 1.0285, 1.6475])
Data_ours = np.array([17.07, 19.58, 21.135, 21.935, 22.305, 22.25, 21.34, 21.845, 21.82, 21.045, 21.385, 21.01])
Data_Blas = np.array([17.885, 20.035, 23.41, 23.815, 21.665, 21.875, 21.01, 21.165, 21.6, 21.415, 22.105, 22.21])

Nl = np.array([31, 32, 64, 96, 97, 127, 128, 129, 255, 256, 257, 511, 512, 513, 
        639, 640, 641, 767, 768, 769, 895, 896, 897, 1023, 1024, 1025]).astype(str)
Starter = np.array([2.372, 2.367, 2.515, 2.4575, 2.4465, 2.518, 2.483, 2.4925, 2.4855, 2.4255, 2.431, 2.514, 2.4385, 
        2.5145, 2.522, 2.49, 2.48, 2.494, 2.299, 2.3715, 2.521, 2.4785, 2.487, 2.4895, 2.4405, 2.4915])
Packing = np.array([2.0855, 2.3065, 2.3925, 2.4165, 2.2485, 2.418, 2.4185, 2.324, 2.4255, 2.4095, 2.382, 2.413, 2.414, 
        2.4035, 2.3985, 2.4295, 2.408, 2.41, 2.43, 2.418, 2.4135, 2.446, 2.424, 2.4155, 2.4415, 2.43])
Unrolling = np.array([2.1615, 2.389, 2.4695, 2.446, 2.292, 2.4505, 2.466, 2.3615, 2.419, 2.475, 2.403, 2.4145, 2.436, 
        2.421, 2.429, 2.452, 2.4235, 2.4295, 2.451, 2.437, 2.4385, 2.462, 2.4285, 2.416, 2.424, 2.422])

K4_4 = np.array([14.475, 16.535, 17.935, 17.545, 16.95, 18.455, 18.44, 16.765, 18.55, 18.285, 17.725, 18.325, 18.03, 
        18.035, 18.355, 18.23, 18.005, 17.775, 17.89, 17.985, 17.79, 18.125, 18.155, 17.155, 17.35, 17.59])
K8_4 = np.array([15.31, 17.345, 18.65, 18.6, 17.02, 18.955, 19.31, 17.685, 19.53, 19.27, 18.42, 19.215, 19.045, 19.11, 
        19.33, 19.49, 19.15, 19.01, 18.975, 19.045, 18.57, 19.07, 18.925, 18.275, 18.73, 19.12])
K16_4 = np.array([15.06, 16.975, 19.425, 19.06, 16.54, 19.595, 19.79, 16.77, 19.915, 19.815, 18.425, 19.535, 19.435, 
        18.75, 19.56, 19.84, 19.055, 19.18, 19.48, 19.165, 18.365, 19.68, 18.965, 18.22, 19.01, 18.765])
"""Data Region }"""

def draw_comparison_plot(X: np.array, Y: list[np.array], 
        fig_name: str, labels: list[str], title: str):
    # create figure
    plt.figure(figsize=(10, 6))
    
    n = len(Y)
    # draw each line (line chart)
    for i in range(n):
        assert X.size >= Y[i].size, "more values than N's"
        plt.plot(X[:Y[i].size], Y[i], label=labels[i])

    # labels, title, light grid, auto legend
    plt.xlabel('Matrix size: N')
    if X.size >= 20:
        plt.xticks(rotation=90)
    plt.ylabel('Performance: GFLOPS')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(f"figures/{fig_name}.png")

if __name__ == "__main__":
    # Q1b
    Y = [Data_naive, Data_ours, Data_Blas]
    labels = ['Naive', 'Ours', 'BLAS']
    title = "Performance Comparison"

    draw_comparison_plot(Ns, Y, "Q1b", labels, title)

    # Q2b Dev Process step-by-step comparison
    # 1. opt tricks
    Y = [Starter, Packing, Unrolling]
    labels = ['Starter', '+=Packing', '+=Unrolling']
    title = "Optimization techniques before changing microkernel"

    draw_comparison_plot(Nl, Y, "opt_tricks", labels, title)

    # 2. with vs without microkernel
    Y = [Unrolling, K4_4]
    labels = ['bl_dgemm_ukr', 'bl_dgemm_444']
    title = "Naive v.s. Vectorized Microkernel"

    draw_comparison_plot(Nl, Y, "naive_vec", labels, title)

    # 3. microkernel comparison
    Y = [K4_4, K8_4, K16_4]
    labels = ['bl_dgemm_444', 'bl_dgemm_844', 'bl_dgemm_1644']
    title = "Vectorized Microkernel Comparison"

    draw_comparison_plot(Nl, Y, "microkernel", labels, title)