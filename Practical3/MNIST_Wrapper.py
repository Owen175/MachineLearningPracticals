import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from CNN import CNN

class MNIST_Wrapper:
    def __init__(self, visualise=False, save=False, load=False, saveFileName=None, loadFileName=None):
        self.save = save
        self.fileName = saveFileName

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [50000, 10000])
        dataset_test = datasets.MNIST('./mnist', train=False, transform=transform)

        self.trainloader = DataLoader(dataset_train, batch_size=64)
        self.testloader = DataLoader(dataset_test, batch_size=64)
        self.validationloader = DataLoader(dataset_valid, batch_size=64)

        if visualise: visualise(dataset_train)

        self.cnn = CNN()

        if load: self.cnn.load_state_dict(torch.load(loadFileName, weights_only=True))
    
    @staticmethod
    def visualise(dataset_train):
        fig = plt.figure()
        for i in range(16):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.set_xticks(())
            ax.set_yticks(())
            ax.imshow(dataset_train[i][0].reshape(28, 28), cmap='Greys_r')
        plt.show()

    ## Training:
    def train(self, epochs=5, lr=1e-4):
        running_count = 0
        running_train_error = 0
        running_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i, (batch_inputs, batch_labels) in enumerate(self.trainloader):
                running_count += 1
                # Zeroes the gradients for the parameters. 
                optimizer.zero_grad()

                # Forwards pass
                outputs = self.cnn(batch_inputs)
                loss = criterion(outputs, batch_labels)

                # Backward pass + optimization
                loss.backward()  # Computes gradients
                optimizer.step()

                # print statistics + update errors
                running_loss += loss.item()
                predicted = _, predicted = torch.max(outputs, 1)
                running_train_error += (predicted == batch_labels).sum().item()
                if running_count % 400 == 0:    # print every 400 mini-batches
                    print("-------------------------")
                    print(f'[{epoch + 1}, {i + 1:5d}] \nloss: {running_loss / 400:.3f}')
                    print(f'training accuracy: {running_train_error/256:.3f}%')
                    print(f'validation accuracy: {self.get_validation_error():.3f}%')
                    # Need to modify to include accuracy on the validation set + training set
                    running_loss = 0
                    running_train_error = 0
        print("-------------------------")
        print("Finished training")

        if self.save: 
            torch.save(self.cnn.state_dict(), './save_state.pt')

    def get_validation_error(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in self.validationloader:
                # calculate outputs by running images through the network
                output = self.cnn(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(output, 1)  # Check if 1 does what we want
                total += len(images)
                correct += (predicted == labels).sum().item()
        return correct/total * 100

    def get_test_error(self):
        # Taken from pytorch docs. 
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for images, labels in self.testloader:
                # calculate outputs by running images through the network
                outputs = self.cnn(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)  # Check if 1 does what we want
                total += len(images)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct / total}%')
        return 100 * correct / total