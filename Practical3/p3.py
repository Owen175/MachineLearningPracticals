import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [50000, 10000])
dataset_test = datasets.MNIST('./mnist', train=False, transform=transform)



# Above code given 
# Imports dataset and normalizes

def visualise():
    fig = plt.figure()
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(dataset_train[i][0].reshape(28, 28), cmap='Greys_r')
    plt.show()


"""
Structure suggested:
Input is Bx1x28x28 - B minibatch size and 1x28x28 is the image in grayscale. 
Initialise weights as gaussian r.v.s with mean 0 and variance 0.0025. Biases as 0.1 (supports ReLU)
Convolutional layer: 25 filters with size 12x12x1 with stride of 2 in both dimensions, no padding. Follow with ReLU
Add another convolutional layer: 64 filters with size 5x5x25 maintaining height and width. Stride 1 and pad when necessary + ReLU
Max pooling with pool size 2x2
flattening operation to transform 2d image into a 1d vector - pytorch flatten
Add fully connected layer with 1024 inputs. ReLU activation
Add dropout layer to avoid overfitting. 
Add another fully connected layer to get 10 output units to softmax for each number. If using cross entropy, softmax not needed in torch
Using cross entropy
Accuracy fraction of data correctly classified. 
Adam optimizer for training - look at docs
Train on minibatches of size 64. 
1000-5000 iterations - start low to see progress
"""

# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Looking at the struture here.
# https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module
# Here for NN

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(12,12), stride=2, padding=0)
        # 1 channel in - grayscale. 25 out as 25 filters. 12x12x1 kernel. 2 stride in both directions. No padding. 
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=(5,5), stride=1, padding=2)
        self.pool = nn.MaxPool2d(2)  # Defines a max pool with a 2x2 square stride 2. 
        self.connected1 = nn.Linear(1024, 1024)  # Unsure if 1024 is the right output size
        self.dropout = nn.Dropout(p=0.2)
        self.connected2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,start_dim=1)  # Don't want the batch to be flattened
        x = F.relu(self.connected1(x))
        x = self.dropout(x)
        x = self.connected2(x)
        return x
## Training:


cnn = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)

## Training:
def train(save=True, epochs=5):
    loader = DataLoader(dataset_train, batch_size=64)

    running_count = 0
    for epoch in range(epochs):
        running_loss = 0
        for i, (batch_inputs, batch_labels) in enumerate(loader):
            running_count += 1
            # Zeroes the gradients for the parameters. 
            optimizer.zero_grad()

            # Forwards pass
            outputs = cnn(batch_inputs)
            loss = criterion(outputs, batch_labels)

            # Backward pass + optimization
            loss.backward()  # Computes gradients
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if running_count % 100 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                # Need to modify to include accuracy on the validation set + training set
                running_loss = 0.0

    print("Finished training")

    if save: 
        torch.save(cnn.state_dict(), './save_state.pt')
    # net.load_state_dict(torch.load(PATH, weights_only=True)) to load

train(epochs=5)


# To predict:
# outputs = cnn(inputs)
# _, predicted = torch.max(outputs, 1)


testloader = DataLoader(dataset_test)


# Taken from pytorch docs. 
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for image, label in testloader:
        # calculate outputs by running images through the network
        output = cnn(image)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(output, 1)  # Check if 1 does what we want
        total += 1
        if predicted == label:
            correct += 1

print(f'Accuracy of the network on the test images: {100 * correct / total} %')
