import torch
import torch.optim as optim
import torch.nn as nn

from data.data_loader import get_data_loaders
from models.model import Net
from utils.logger import progress_bar
from utils.visualisations import plot_loss

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-7)
    EPOCHS = 4

    trainloader, testloader, _ = get_data_loaders(batch_size=8)

    losses = []

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        step_count = 0
        for i, train_data in enumerate(trainloader):
            inputs, labels = train_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            step_count += len(trainloader) 

            losses.append(loss.item())

            progress_bar(epoch * len(trainloader) + i, EPOCHS * len(trainloader), prefix='Training Progress:', suffix=f'Epoch {epoch+1}/{EPOCHS} Batch {i}/{len(trainloader)} Loss: {loss.item()}', length=50)

    plot_loss(losses)
    torch.save(net.state_dict(), './models/model.pth')

