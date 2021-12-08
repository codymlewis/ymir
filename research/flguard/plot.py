import matplotlib.pyplot as plt
import pickle

import numpy as np
from absl import app


def main(_):
    exts = ['10e', '10e_0.01lr', '30e', '30e_0.01lr', '50e_0.01lr', '100e_0.01lr', 'sgd', 'sgd_0.01lr', 'sgd_128b', 'sgd_128b_0.01lr', 'rande1-10', 'rande10-20', 'rande20-30']
    fig, axs = plt.subplots(round(len(exts)**0.5), round(len(exts)**0.5), figsize=(10, 10))
    for ext, ax in zip(exts, axs.flatten()):
        with open(f"results.{ext}.pkl", 'rb') as f:
            plot_accuracy(f"{ext}", pickle.load(f), ax)
    axs[0, 0].legend()
    plot_epoch_bar("rande1-10_dist", axs[3, 1], [6, 9, 5, 3, 6, 5, 10, 4, 7, 5])
    plot_epoch_bar("rande10-20_dist", axs[3, 2], [15, 19, 15, 12, 15, 14, 20, 13, 16, 15])
    plot_epoch_bar("rande20-30_dist", axs[3, 3], [25, 29, 25, 22, 25, 24, 30, 23, 26, 25])
    fig.tight_layout()
    filename = "plot.png"
    print(f"Saving plot to {filename}")
    plt.savefig(filename, dpi=320)


def accuracy(M):
    return np.diagonal(M, axis1=1, axis2=2).sum(axis=1) / M.sum(axis=(1, 2))


def plot_accuracy(title, data, ax):
    ax.plot(accuracy(data['train']), label='train')
    ax.plot(accuracy(data['test']), label='test')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Round')
    # print(f"Saving plot to {filename}")
    # plt.savefig(filename, dpi=320)


def plot_epoch_bar(title, ax, epochs):
    ax.bar(range(len(epochs)), epochs)
    ax.set_xticks(range(len(epochs)))
    ax.set_title(title)
    ax.set_ylabel('Epoch')
    ax.set_xlabel('Endpoint id')

if __name__ == '__main__':
    app.run(main)