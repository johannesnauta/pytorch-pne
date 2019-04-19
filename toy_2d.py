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
    epochs = 20000
    learning_rate = 1e-3
    training_samples = 1000
    validation_samples = training_samples
    test_samples = 100*100
    batch_size = 16
    measurements = epochs//200  # Measure every n steps

    # Define network architecture
    input_dim = 2
    output_dim = 2
    # Define systems
    # current available data sets: SimpleTwoDimensional(N, seed)
    datamic_systems = ["linear"]

    seeds = np.random.randint(1e4, size=ensemble_size)
    # Allocate for losses
    train_error = np.zeros((ensemble_size, measurements))
    validation_error = np.zeros(train_error.shape)
    test_error = np.zeros(train_error.shape)
    
    for system in datamic_systems:
        # Instantiate class objects
        # data = datamics.datamics(training_samples, system)
        data = src.dataloader.SimpleTwoDimensional(seed=size)
        # Initialize models
        ensemble = [pnn.Model(input_dim, output_dim, learning_rate, seeds[i]) for i in range(ensemble_size)]
        # Initialize & allocate
        ensemble_mean = np.zeros((test_samples, 1))
        ensemble_var = np.zeros((test_samples, 1))

        # Get training data 
        data.gather_train_samples(training_samples)
        data.gather_validation_samples(validation_samples)
        data.gather_test_samples(test_samples)

        # Train an ensemble of probabilistic networks:
        i = 0
        for model in ensemble:
            j = 0
            for epoch in range(epochs):
                minibatch = data.get_batch(batch_size)
                model_in = minibatch[:,:2]
                y = minibatch[:,-1]
                model.step(model_in, y)
                if (epoch+1)%(epochs//measurements)==0:
                    # Compute error on train, test and validation sets
                    train_error[i,j], validation_error[i,j], test_error[i,j] = model.compute_errors(data.training_data, data.validation_data, data.test_data)
                    j += 1
                if boss and (epoch+1)%100 == 0:
                    print("Training model %s/%s, epoch %s/%s"%(size*(i+1), size*ensemble_size, epoch+1, epochs), end='\r')

            # Test model on training samples
            mean, var = model.forward(data.test_data[:,:2], "nparray")

            # Add to ensemble mean and variance
            ensemble_mean += mean 
            ensemble_var += var + mean**2
             
            i += 1
        
        # Gather data from all cores
        gathered_means = comm.gather(ensemble_mean)
        gathered_vars = comm.gather(ensemble_var)
        gathered_train_err = comm.gather(train_error)
        gathered_val_err = comm.gather(validation_error)
        gathered_test_err = comm.gather(test_error)
        if boss:
            # Compute ensemble mean and averages
            stacked_means = np.reshape(np.stack(gathered_means, axis=0),(size, test_samples))
            stacked_vars = np.reshape(np.stack(gathered_vars, axis=0),(size, test_samples))
            assert(stacked_means.shape == stacked_vars.shape)
            ensemble_mean = np.sum(stacked_means, axis=0) / (size*ensemble_size)
            ensemble_var = np.sum(stacked_vars, axis=0) / (size*ensemble_size)
            ensemble_var = ensemble_var - ensemble_mean**2
            # Save ensemble
            np.save("data/ensemble_mean_%s"%(system), ensemble_mean)
            np.save("data/ensemble_var_%s"%(system), ensemble_var)
            # Compute mean errors
            stacked_train_err = np.stack(gathered_train_err, axis=0)
            mean_train_err = np.mean(np.mean(gathered_train_err, axis=0), axis=0)
            np.save("data/train_error_%s"%(system), mean_train_err)
            stacked_val_err = np.stack(gathered_val_err, axis=0)
            mean_val_err = np.mean(np.mean(gathered_val_err, axis=0), axis=0)
            np.save("data/val_error_%s"%(system), mean_val_err)
            stacked_test_err = np.stack(gathered_test_err, axis=0)
            mean_test_err = np.mean(np.mean(gathered_test_err, axis=0), axis=0)
            np.save("data/test_error_%s"%(system), mean_test_err)


            # Save things for plotting
            np.save("data/test_samples", data.test_data)
            print("\n...")
    if boss:
        print("\nComputation time: %.2fs"%(time.time()-starttime))

        