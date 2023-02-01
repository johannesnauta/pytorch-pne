""" Plot data from the data directory """
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import src.dataloader
class Plotter():
    def __init__(self):
        self.data = src.dataloader.TransitionData(seed=42, prefix='data_in/acrobot_')
        self.data.gather_test_samples(100**2, 10)


    def plot_ensemble(self, ax):
        test_data = np.load("data/test_samples.npy")
        mean = np.load("data/ensemble_mean_linear.npy")
        var = np.load("data/ensemble_var_linear.npy")

        states = test_data[: , 0:self.data.state_dim]
        actions = test_data[: , self.data.state_dim:(self.data.state_dim+self.data.action_dim)]
        output = test_data[: , self.data.input_dim:(self.data.input_dim+self.data.output_dim)]

        #  nsamples = len(output)
        #  N = int(np.sqrt(nsamples))
        #  output = output.reshape((N,N))
        #  mean = mean.reshape((N,N), order='C')
        #  var = var.reshape((N,N), order='C')

        extent = [min(self.data.xaxis), max(self.data.xaxis), min(self.data.yaxis), max(self.data.yaxis)]
        aspect = (extent[1]-extent[0]) / (extent[3]-extent[2])
        truth_im = ax[0].imshow(np.rot90(output, axes=(0,1)), extent=extent, aspect=aspect)

        ax[0].set_title("ground truth")
        mean_im = ax[1].imshow(np.rot90(mean, axes=(0,1)), extent=extent, aspect=aspect, vmin=-1.1, vmax=1.1)
        ax[1].set_title(r"$\mu_{ensemble}$")
        std_im = ax[2].imshow(np.rot90(np.sqrt(var), axes=(0,1)), extent=extent, aspect=aspect, norm=LogNorm(vmin=0.005, vmax=3))
        ax[2].set_title(r"$\sigma_{ensemble}$")

        for a in ax:
            a.set_xlabel(r'x')
            a.set_ylabel(r'y')

    def plot_uncertainty(self, ax):
        # Gather training and test self.data sets
        test_data = np.load("data/test_samples.npy")
        var = np.load("data/ensemble_var.npy")
        test_data_idx = np.load("data/test_data_idx.npy")
        # ax[0].scatter(test_data[:,0], test_data[:,1], color='b', s=0.4, label='test self.data')
        ax[0].scatter(self.data.training_data[:,0], self.data.training_data[:,1], color='r', s=0.4, label='training self.data')

        #  ax[0].contour(self.data.xaxis, self.data.yaxis, std.T, levels=8)

        extent = [min(self.data.xaxis), max(self.data.xaxis), min(self.data.yaxis), max(self.data.yaxis)]
        aspect = (extent[1]-extent[0]) / (extent[3]-extent[2])

        # Construct image of variance
        N = int(np.sqrt(len(var)/10))
        image = np.zeros((N,N))
        print("image shape", image.shape)
        for ((xi, yi), var_val) in zip(test_data_idx, var):
            image[xi, yi] += np.mean(var_val)/10
            # TODO normalize?
        print("MEAN: ", np.mean(image))


        im = ax[1].imshow(np.rot90(image, axes=(0,1)), extent=extent, aspect=aspect, interpolation='spline16', norm=LogNorm(vmin=0.01, vmax=10))
        plt.colorbar(im, ax=ax[1])

        ax[1].set_xlabel(r'$x$')
        ax[1].set_ylabel(r'$a$')
        ax[1].set_title(r'$\sigma$')

        ax[0].legend()


if __name__ == "__main__":
    pjotr = Plotter()

    # Plot mean
    #  meanfig, meanax = plt.subplots(1,3, figsize=(18,6))
    #  pjotr.plot_ensemble(meanax)

    # Plot state space coverage of training and test self.data
    covfig, covax = plt.subplots(1,2, figsize=(12,6))
    pjotr.plot_uncertainty(covax)

    if "save" in sys.argv:
        meanfig.savefig("figures/2d_regression")
        covfig.savefig("figures/2d_std")
    else:
        plt.show()
