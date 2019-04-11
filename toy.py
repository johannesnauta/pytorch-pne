""" Toy function which wishes to fit n bootstraps on data points from a dataset """
# Import libraries
import sys 
import time 
import torch
import numpy as np 
# Initialize MPI for parallelisation
from mpi4py import MPI 
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
boss = rank==0

import src.dataloader
import models.pnn as models

if __name__ == "__main__":
    starttime = time.time()
    # Parameters
    ensemble_size = 2       # Ensemble size per core
    batch_size = 10         
    epochs = 5000
    data_samples = 1000
    test_samples = 500      
    decay_rate = 100        # Learning rate decay

    # Initialize objects
    # available data sets: Sinewave(N, seed), Simplecurve(N, seed)
    datasets = ['sinewave', 'simplecurve']
    for curve in datasets:
        if boss:
            print("\nLearning %s..."%(curve))
        if curve == 'sinewave':
            dataloader = src.dataloader.Sinewave(N=data_samples, seed=size)
        if curve == 'simplecurve':
            dataloader = src.dataloader.Simplecurve(N=data_samples, seed=size)
        # Initialize ensemble of networks
        ensemble = [models.Model((i+1)*rank) for i in range(ensemble_size)]
        ensemble_mean = 0
        ensemble_var = 0
        means = []
        variances = []

        # Generate test data set
        xtest, ytest = dataloader.get_test_samples(N=test_samples)
        i = 0
        for model in ensemble:
            for epoch in range(epochs):
                x, y = dataloader.get_samples(batch_size)
                # Train model
                model.step(x,y)
                if (epoch+1) % decay_rate == 0:
                    # Adjust learning rate every n steps
                    model.adjust_learning_rate()
                    if boss:  
                        print("Training models %i/%i, epochs %i/%i"%(size*(i+1), size*ensemble_size, epoch+1,epochs), end='\r')

            # Test model
            mean, var = model.forward(xtest)
            # Add to the ensemble mean and variance
            ensemble_mean += mean
            ensemble_var += var + mean**2
            # Save outputs of each network
            means.append(mean)
            variances.append(var)
            # Increment
            i += 1

        # Store the seperate means and variances
        np.save("data/means_%s_%i"%(dataloader.name, rank), means)
        np.save("data/variances_%s_%i"%(dataloader.name, rank), variances)

        # Gather all data from all cores
        gathered_means = comm.gather(means)
        gathered_variances = comm.gather(variances)
        if boss:
            ensemble_mean = np.zeros(test_samples)
            ensemble_var = np.zeros(test_samples)
            # Compute ensemble mean and variance across all cores 
            stacked_means = np.reshape(np.stack(gathered_means, axis=0), (size*ensemble_size, test_samples))
            stacked_vars = np.reshape(np.stack(gathered_variances, axis=0), (size*ensemble_size, test_samples))
            assert(stacked_means.shape == stacked_vars.shape)
            
            ensemble_mean = np.mean(stacked_means, axis=0)
            ensemble_var = np.sum(stacked_vars + stacked_means**2, axis=0) / (size*ensemble_size)
            ensemble_var = ensemble_var - ensemble_mean**2

            # Save data
            np.save("data/ensemble_mean_%s"%(dataloader.name), ensemble_mean)
            np.save("data/ensemble_var_%s"%(dataloader.name), ensemble_var)
            np.save("data/means", means)
            np.save("data/variances", variances)
    if boss:
        print("\nComputation time: %.2fs"%(time.time()-starttime))




    
