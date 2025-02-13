import torch
import torchvision
import torchvision.transforms as transforms
import os

def get_data_loaders(batch_size = 4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define the paths
    train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/Train_Test_Valid/Train'))
    test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/Train_Test_Valid/test'))
    # Load the training data
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Load the test data
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = ('barrel', 'blue', 'compass', 'lions_mane', 'mauve','moon')

    return trainloader, testloader, classes