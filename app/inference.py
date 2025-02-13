from PIL import Image
import torch
import torchvision.transforms as transforms

from models.model import Net

def classify_jellyfish(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    classes = ('barrel', 'blue', 'compass', 'lions_mane', 'mauve','moon')

    net = Net()
    net.load_state_dict(torch.load('./models/model.pth'))
    net.to(device)

    output = net(image)
    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]