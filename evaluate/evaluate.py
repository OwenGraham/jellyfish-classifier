import torch

from models.model import Net
from data.data_loader import get_data_loaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.load_state_dict(torch.load('./models/model.pth'))
net.to(device)

net.eval()

_,testloader,_=get_data_loaders()

correct = 0
total = 0
with(torch.no_grad()):
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))