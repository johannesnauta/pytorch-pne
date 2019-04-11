import sys
import numpy as np 
import matplotlib.pyplot as plt 

import src.dataloader

class Plotter():
    def __init__(self, dataset):
        # available data sets: Sinewave(N, seed), Simplecurve(N, seed)
        if dataset == 'sinewave':
            self.dload = src.dataloader.Sinewave(N=500, seed=42)
        if dataset == 'simplecurve':
            self.dload = src.dataloader.Simplecurve(N=500, seed=42)

    def plot_ensemble(self, ax, mean, var):
        mean = mean.flatten()
        # mean = mean * self.dload.input_std + self.dload.input_mean
        std = np.sqrt(var.flatten())
        xax, y = self.dload.get_test_samples(N=len(mean))
        # Plot predicted mean
        ax.plot(xax, mean, color='k', label='ensemble')
        ax.fill_between(xax, mean-std, mean+std, color='gray', alpha=0.5)
        ax.fill_between(xax, mean-2*std, mean+2*std, color='gray', alpha=0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    def plot_seperate_models(self, ax, means, variances):
        xax = self.dload.x * self.dload.input_std + self.dload.input_mean
        n_cores = len(means)
        assert(n_cores==len(variances))
        for n in range(n_cores):
            for i in range(len(means[n])):
                mean = means[n][i,:,0]
                std = np.sqrt(variances[n][i,:,0])
                ax.plot(xax, mean)
                ax.fill_between(xax, mean-std, mean+std, alpha=0.5)

    def plot_training_data(self, ax):
        trainx = self.dload.x 
        trainy = self.dload.y
        ax.scatter(trainx, trainy, marker='x', s=10, color='g', label='sample training data')

    def plot_truth(self, ax, N):
        x, y = self.dload.get_test_samples(N=N)
        ax.plot(x, y, '--r', label='ground truth')

if __name__ == "__main__":    
    datasets = ['sinewave', 'simplecurve']
    fig, headax = plt.subplots(1,2, figsize=(2*6,6))
    axi = 0
    for dataset in datasets:
        pjotr = Plotter(dataset)
        # Load data
        en_mean = np.load("data/ensemble_mean_%s.npy"%(pjotr.dload.name))
        en_var = np.load("data/ensemble_var_%s.npy"%(pjotr.dload.name))
        means = np.load("data/means.npy")
        variances = np.load("data/variances.npy")
        # Generate plots        
        ax = headax[axi]
        pjotr.plot_truth(ax, len(en_mean.flatten()))
        pjotr.plot_training_data(ax)

        means = np.empty(4, dtype=object)
        variances = np.empty(means.shape, dtype=object)
        for nc in range(4):
            means[nc] = np.load("data/means_%s_%i.npy"%(pjotr.dload.name, nc))
            variances[nc] = np.load("data/variances_%s_%i.npy"%(pjotr.dload.name, nc))
        pjotr.plot_ensemble(ax, en_mean, en_var)
        # Additionally, one can plot the seperate models
        # pjotr.plot_seperate_models(ax, means, variances)

        # Finish
        ax.legend(loc='upper left')
        axi += 1
    
    # Save or show
    if "save" in sys.argv:
        fig.savefig("figures/results", bbox_inches='tight')
    else:
        plt.show()

