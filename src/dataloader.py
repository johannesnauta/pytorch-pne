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
        f = 1
        xl = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin 
        xr = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin + (f-1) * (self.xmax - self.xmin) / f
        self.x = np.concatenate((xl, xr), axis=0)
        scale = 0.0225 * np.abs(np.sin(1.5 * self.x + np.pi / 8))
        noise = np.random.normal(0, scale=scale, size=len(self.x))
        self.y = np.sin(self.x) + noise

        # Standardize data
        # self.input_mean = np.mean(self.x)
        # self.input_std = np.std(self.x)
        # self.x = (self.x-self.input_mean)/self.input_std

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
        # x = (x-self.input_mean)/self.input_std 
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
        f = 1
        xl = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin 
        xr = (self.xmax - self.xmin) / f * np.random.random(size=N//2) + self.xmin + (f-1) * (self.xmax - self.xmin) / f
        self.x = np.concatenate((xl, xr), axis=0)
        noise = np.random.normal(0, 0.05*self.xmax**2, size=len(self.x))
        self.y = self.x**2 + noise 
        # Standardize data
        # self.input_mean = np.mean(self.x)
        # self.input_std = np.std(self.x)
        # self.x = (self.x-self.input_mean)/self.input_std

    def get_samples(self, N):
        """ Get N samples from the generated dataset """
        indices = np.random.choice(np.arange(len(self.x)), size=N)
        x = self.x[indices]
        y = self.y[indices]
        return x, y

    def get_test_samples(self, N):
        a = 0.5
        x = np.linspace(self.xmin-a*(self.xmax-self.xmin), self.xmax+a*(self.xmax-self.xmin), num=N)
        y = x**2
        # Standardize data
        # x = (x-self.input_mean)/self.input_std 
        return x, y

class SimpleTwoDimensional():
    """ Generate two-dimensional points from a simple problem
        in this case: z = cos(x)sin(y)
    """
    def __init__(self, seed):
        self.x_size = np.pi 
        self.y_size = np.pi
        np.random.seed(seed)

    def gather_train_samples(self, n_samples):
        self.training_data = self.uniform_sampling(n_samples)
        self.ntrain_samples = n_samples

    def gather_validation_samples(self, n_samples):
        self.validation_data = self.uniform_sampling(n_samples)

    def gather_test_samples(self, n_samples):
        # self.test_data = self.uniform_sampling(n_samples, self.action_size)
        N = int(np.sqrt(n_samples))
        x = np.linspace(-self.x_size, self.x_size, num=N)
        y = np.linspace(-self.y_size, self.y_size, num=N)
        self.test_data = np.zeros((n_samples, 3))
        i = 0
        for xi in x:
            for yi in y:
                self.test_data[i] = np.array([xi, yi, self.system(xi, yi)])
                i += 1
        
        self.xaxis = x[:]
        self.yaxis = y[:]

    def get_batch(self, batch_size):
        """ Sample batch_size (state,action)-pairs from the training data """
        indices = np.random.choice(range(self.ntrain_samples-1),size=batch_size,replace=False)
        return self.training_data[indices]

    def uniform_sampling(self, n_samples):
        """ Uniformly sample (state,action) pairs for model learning 
            - replaces standard dynamical model for testing purposes
        """
        # Sample uniformly from within the domain
        # x = np.random.uniform(-self.x_size, self.x_size, size=n_samples)
        # y = np.random.uniform(self.y_size, self.y_size, size=n_samples)

        # Generate two clusters within the 2D state space
        x = np.random.uniform(-self.y_size, 0, size=n_samples//2)
        x = np.concatenate((x, np.random.uniform(0, self.x_size, size=n_samples//2)), axis=0)
        y = np.random.uniform(-self.y_size, 0, size=n_samples//2)
        y = np.concatenate((y, np.random.uniform(0, self.y_size, size=n_samples//2)))
        outputs = self.system(x, y)
        data = np.stack((x, y, outputs), axis=1)

        return data 

    def system(self, x, y):
        """ Define system evolution """
        output = np.sin(x) * np.sin(y)
        return output