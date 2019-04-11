""" Generate data both for training and testing """
# Import necessary libraries
import numpy as np 
# import matplotlib.pyplot as plt 

class Sinewave():
    """ Generate points from a given sine wave """
    def __init__(self, N, seed):
        self.xmin = -2*np.pi
        self.xmax = 2*np.pi
        np.random.seed(seed) 
        self.name = "sinewave"

        # Divide interval in f parts and generate samples from the first and last
        f = 4
        xl = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin 
        xr = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin + (f-1) * (self.xmax - self.xmin) / f
        self.x = np.concatenate((xl, xr), axis=0)
        scale = 0.0225 * np.abs(np.sin(1.5 * self.x + np.pi / 8))
        noise = np.random.normal(0, scale=scale, size=len(self.x))
        self.y = np.sin(self.x) + noise

        # Standardize data
        self.input_mean = np.mean(self.x)
        self.input_std = np.std(self.x)
        self.x = (self.x-self.input_mean)/self.input_std

    def get_samples(self, N):
        """ Get N samples from the generated dataset """
        indices = np.random.choice(np.arange(len(self.x)), size=N)
        x = self.x[indices]
        y = self.y[indices]
        return x, y
    
    def get_test_samples(self, N):
        a = 0.5
        delta = self.xmax - self.xmin 
        x = np.linspace(self.xmin-a*delta, self.xmax+a*delta, num=N)
        y = np.sin(x)
        # Standardize data
        x = (x-self.input_mean)/self.input_std 
        return x, y

class Simplecurve():
    """ Generate points from a simple curve
        in this case, y=x**3
    """
    def __init__(self, N, seed):
        self.xmax = -2
        self.xmin = 2
        np.random.seed(seed)
        self.name = "simplecurve"
        
        # Divide interval in f parts and generate samples from the first and last
        f = 3
        xl = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin 
        xr = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin + (f-1) * (self.xmax - self.xmin) / f
        self.x = np.concatenate((xl, xr), axis=0)
        noise = np.random.normal(0, 0.25*self.xmax**2, size=len(self.x))
        self.y = self.x**3 + noise 
        # Standardize data
        self.input_mean = np.mean(self.x)
        self.input_std = np.std(self.x)
        self.x = (self.x-self.input_mean)/self.input_std

    def get_samples(self, N):
        """ Get N samples from the generated dataset """
        indices = np.random.choice(np.arange(len(self.x)), size=N)
        x = self.x[indices]
        y = self.y[indices]
        return x, y

    def get_test_samples(self, N):
        a = 0.5
        x = np.linspace(self.xmin-a*(self.xmax-self.xmin), self.xmax+a*(self.xmax-self.xmin), num=N)
        y = x**3
        # Standardize data
        x = (x-self.input_mean)/self.input_std 
        return x, y

