""" Probabilistic Neural Network
    outputs means and variances given an input which define a 
    distribution from which the next state can be sampled
"""
# Import necessary libraries
import numpy as np 
import torch 

class Model():
    def __init__(self, input_dim, output_dim, learning_rate, seed):
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = 32
        # Instantiate model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.hidden_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, self.output_dim, bias=True)
        ).to(torch.device('cpu'))
        # Instantiate optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.999

    def softplus(self, x):
        """ Compute softplus """
        softplus = torch.log(1+torch.exp(x))
        # Avoid infinities due to taking the exponent
        softplus = torch.where(softplus==float('inf'), x, softplus)
        return softplus

    def NLL(self, mean, var, truth):
        """ Compute the negative log likelihood """
        diff = torch.sub(truth, mean)
        var = self.softplus(var)
        # Compute loss 
        loss = torch.mean(torch.div(diff**2, 2*var))
        loss += torch.mean(0.5*torch.log(var))
        return loss.sum()

    def forward(self, inputs, input_type):
        """ Forward pass for a given input
            :input:     (state,action)-pair
            :returns:   means and variances given input
        """
        # Compute output of model
        if input_type == "nparray":            
            x = torch.from_numpy(inputs).float()
            out = self.model(x)
            mean, var = torch.split(out, self.output_dim//2, dim=1)
            var = self.softplus(var)
            return mean.detach().numpy(), var.detach().numpy()
        else:
            out = self.model(inputs)
            mean, var = torch.split(out, self.output_dim//2, dim=1)
            var = self.softplus(var)
            return mean, var
        
    def step(self, inputs, true_out):
        """ Execute gradient step given the samples in the minibatch """
        # Convert input and true_out to useable tensors
        x = torch.from_numpy(inputs).float()
        y = torch.from_numpy(true_out).unsqueeze(-1).float()

        # Compute output of model
        out = self.model(x)
        mean, var = torch.split(out, self.output_dim//2, dim=1)
        # Compute loss 
        self.nll = self.NLL(mean, var, y)

        # Backpropagate the loss
        self.optimizer.zero_grad()
        self.nll.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        self.adjust_learning_rate()

    def compute_errors(self, train_in, validation_in, test_in):
        """ Compute loss on the training, validation and test data """
        # Training data
        train_in = torch.from_numpy(train_in).float()
        train_in, train_out = torch.split(train_in, [2,1], dim=1)
        mean, var = self.forward(train_in, "tensor")
        train_loss = self.NLL(mean, var, train_out).item()

        # Validation data
        validation_in = torch.from_numpy(validation_in).float()
        validation_in, val_out = torch.split(validation_in, [2,1], dim=1)
        mean, var = self.forward(validation_in, "tensor")
        val_loss = self.NLL(mean, var, val_out).item()

        # Test data
        test_in = torch.from_numpy(test_in).float()
        test_in, test_out = torch.split(test_in, [2,1], dim=1)
        mean, var = self.forward(test_in, "tensor")
        test_loss = self.NLL(mean, var, test_out).item()

        return train_loss, val_loss, test_loss