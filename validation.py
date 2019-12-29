import os

import torch
import torchvision
from torchvision import transforms
import torch.utils.data
from mrdataset import BrainsDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './data'
BATCH_SIZE = 128

MODEL_PATH = './models'
MODEL_NAME = 'vgg19_bn.pth'

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.RandomCrop(36, padding=4),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load data



def main():
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    #dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    #    transforms.ToTensor(),
    #    normalize,
    #]))

    dataset = BrainsDataset("data/datacv2", 32)
    #dataset_loader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=128, shuffle=True,
    #    num_workers=4, pin_memory=True)
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True)

    print("Val numbers:%d"%(len(dataset)))

    # Load model
    if device.type == 'cuda':
        model = torch.load(MODEL_PATH + '/' + MODEL_NAME).to(device)
    else:
        model = torch.load(MODEL_PATH + '/' + MODEL_NAME, map_location='cpu')
    model.eval()
    #test
    #test2
    correct = 0.
    total = 0
    i=0
    for images, labels in dataset_loader:
        i=i+1
        # to GPU
        images = images.to(device)
        labels = labels.to(device)
        #print(labels)
        # print prediction
        outputs = model(images)
        # equal prediction and acc
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        # val_loader total
        total += labels.size(0)
        # add correct
        correct += (predicted == labels).sum().item()
        if i>2:
            break

    print("Acc:%.4f."%(correct / total))

#test
if __name__ == '__main__':
    main()
