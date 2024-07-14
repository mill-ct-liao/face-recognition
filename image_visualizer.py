import matplotlib.pyplot as plt
import os

# Visualize distributions
def visualize_distribution(data, title, xlabel, ylabel, save_path=None):
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()