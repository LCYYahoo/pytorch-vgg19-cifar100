import os

import time

import torch
import torchvision
from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset

# first train run this code
from vgg import vgg19_bn
# incremental training comments out that line of code.

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WORK_DIR = './data'
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

MODEL_PATH = './models'
MODEL_NAME = 'vgg19_bn.pth'



# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

class MyDataset(Dataset):
def __init__(self, txt_path, transform = None, target_transform = None):
	fh = open(txt_path, 'r')
	imgs = []
	for line in fh:
		line = line.rstrip()
		words = line.split()
		imgs.append((words[0], int(words[1])))
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
def __getitem__(self, index):
	fn, label = self.imgs[index]
	img = Image.open(fn).convert('RGB')
	if self.transform is not None:
		img = self.transform(img)
	return img, label
def __len__(self):
	return len(self.imgs)



def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.data(root='./data/datacv', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(1200,interpolation=2),
        transforms.Pad(4,padding_mode='edge'),
        transforms.ToTensor(),
        normalize,
    ]))

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)

    print("Train numbers:%d"%(len(dataset)))

    # first train run this line
    model = vgg19_bn().to(device)
    # Load model
    # if device == 'cuda':
    #     model = torch.load(MODEL_PATH + MODEL_NAME).to(device)
    # else:
    #     model = torch.load(MODEL_PATH + MODEL_NAME, map_location='cpu')
    # cast
    cast = torch.nn.CrossEntropyLoss().to(device)
    # Optimization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-8)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        # cal one epoch time
        start = time.time()

        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Step %f"%((step * BATCH_SIZE)/(NUM_EPOCHS * len(dataset)))+" loss %f"%(loss.item()))

            step += 1

        # cal train one epoch time
        end = time.time()
        print("Epoch %f time: {%f - %f} sec!"%((epoch)/(NUM_EPOCHS),end,start))

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + '/' + MODEL_NAME)
    print("Model save to %s"%(MODEL_PATH + '/' + MODEL_NAME))


if __name__ == '__main__':
    main()
