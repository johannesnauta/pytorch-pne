""" Predict the next state given an input (state,action)-pair using a PNE """
# Import libraries
import numpy as np
import sys
import time
# MPI for parallelisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
boss = rank==0
# Seed RNG
np.random.seed(rank)

# Import modules
import src.dataloader
import models.pnn_2d as pnn

if __name__ == "__main__":
    starttime = time.time()
    # Define parameters
    ensemble_size = 2           # Ensemble size per core
    epochs = 1000
    learning_rate = 1e-3
    batch_size = 16
    measurements = epochs//200  # Measure every n steps

    seeds = np.random.randint(1e4, size=ensemble_size)

    data = src.dataloader.TransitionData(seed=size, prefix='data_in/acrobot_')
    data.gather_test_samples(100**2, 10)
    n_test_samples = len(data.test_data)
    # Initialize models
    ensemble = [pnn.Model(data.input_dim, data.output_dim*2, learning_rate, seeds[i]) for i in range(ensemble_size)]
    # Initialize & allocate
    ensemble_mean = np.zeros((n_test_samples, data.output_dim))
    ensemble_var = np.zeros((n_test_samples, data.output_dim))


    # Train an ensemble of probabilistic networks:
    i = 0
    for model in ensemble:
        j = 0
        for epoch in range(epochs):
            minibatch = data.get_batch(batch_size)
            model_in = minibatch[:,:data.input_dim]
            y = minibatch[:,data.input_dim:data.input_dim+model.output_dim]
            #  print(model_in)
            #  print(y)
            model.step(model_in, y)
            if (epoch+1)%(epochs//measurements)==0:
                j += 1
            if boss and (epoch+1)%100 == 0:
                print("Training model %s/%s, epoch %s/%s"%(size*(i+1), size*ensemble_size, epoch+1, epochs), end='\r')

        # Test model on training samples
        mean, var = model.forward(data.test_data[:,:data.input_dim], "nparray")

        # Add to ensemble mean and variance
        ensemble_mean += mean
        ensemble_var += var + mean**2

        i += 1

    # Gather data from all cores
    gathered_means = comm.gather(ensemble_mean)
    gathered_vars = comm.gather(ensemble_var)
    if boss:
        print("gathered_means:",gathered_means)
        print(np.stack(gathered_means, axis=0).shape)
        print(gathered_vars)
        print(np.stack(gathered_vars, axis=0).shape)
        print("SIZE: " , size)

        # Compute ensemble mean and averages
        stacked_means = np.stack(gathered_means, axis=0)
        stacked_vars = np.stack(gathered_vars, axis=0)
        assert(stacked_means.shape == stacked_vars.shape)
        ensemble_mean = np.sum(stacked_means, axis=0) / (size*ensemble_size)
        ensemble_var = np.sum(stacked_vars, axis=0) / (size*ensemble_size)
        ensemble_var = ensemble_var - ensemble_mean**2
        print("final ensemble mean shape:",ensemble_mean.shape)
        # Save ensemble
        np.save("data/ensemble_mean", ensemble_mean)
        np.save("data/ensemble_var", ensemble_var)


        # Save things for plotting
        np.save("data/test_samples", data.test_data)
        np.save("data/test_data_idx", data.test_data_idx)
        print("\n...")
        print("\nComputation time: %.2fs"%(time.time()-starttime))

