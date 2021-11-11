""" Model classes defined here! """

import torch
import torch.nn.functional as F
import torchvision

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(784, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(in_channels=n1_chan, out_channels=10, kernel_size=n2_kern)
        maxpoolksize = 30 - (n1_kern + n2_kern)
        self.pool = torch.nn.MaxPool2d(kernel_size=maxpoolksize)
        
        
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = x.view(-1, 1, 28, 28)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.reshape(-1, 10)
        #x = self.fc1(x)        
        return x

class BestNN(torch.nn.Module):
    # take hyperparameters from the command line args!
    def __init__(self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                 n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features):
        super(BestNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n1_channels, kernel_size=n1_kernel)
        self.conv2 = torch.nn.Conv2d(in_channels=n1_channels, out_channels=n2_channels, kernel_size=n2_kernel)
        #maxpoolksize1 = 30- (n1_kernel + n2_kernel)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = pool1)
        self.conv3 = torch.nn.Conv2d(in_channels=n2_channels, out_channels=n3_channels, kernel_size=n3_kernel)
        self.conv4 = torch.nn.Conv2d(in_channels=n3_channels, out_channels=n4_channels, kernel_size=n4_kernel)
        #maxpoolksize2 = maxpoolksize1 - (n3_kernel + n4_kernel)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=pool2)
        linear_in = 30 - (n1_kernel + n2_kernel)
        linear_in = linear_in//pool1
        linear_in += -(n3_kernel + n4_kernel) + 2
        linear_in = linear_in//pool2
        #self.changeshape = torch.Tensor.view(-1, linear_in**2*n4_channels)
        self.fc1 = torch.nn.Linear(in_features=linear_in**2*n4_channels, out_features=linear_features)
        self.fc2 = torch.nn.Linear(in_features=linear_features, out_features=10)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        #print(f'Reshaped: {x.shape}')
        x = self.hflip(x)   #data augmentation
        #print(f'Flipped: {x.shape}')
        x = torch.relu(self.conv1(x))
        #print(f'Conv1: {x.shape}')
        x = torch.relu(self.conv2(x))
        #print(f'Conv2: {x.shape}')
        x = self.pool1(x)
        #print(f'Pool1: {x.shape}')
        x = torch.relu(self.conv3(x))
        #print(f'Conv3: {x.shape}')
        x = torch.relu(self.conv4(x))
        #print(f'Conv4: {x.shape}')
        x = self.pool2(x)
        #print(f'Pool2: {x.shape}')
        #x = self.changeshape(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print(f'Fc1: {x.shape}')
        x = self.fc2(x)
        return x
