import torch
import os
import pandas as pd
from network import snn
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # tries to put tensors on cuda, if cuda is available

class dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.pairs_frame = pd.read_csv(csv_file, header=None, names=['img1', 'img2', 'label'])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 0])
        img2_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 1])
        img1 = Image.open(img1_name).convert("L")  # convert to grayscale
        img2 = Image.open(img2_name).convert("L")
        label = self.pairs_frame.iloc[idx, 2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

training_data = dataset(csv_file='sign_data/train_data.csv', root_dir='sign_data/Dataset/train/', transform=transform)
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)



print(training_data[0])




# random_data1 = torch.rand(1, 1, 28, 28).to(device)
# random_data2 = torch.rand(1, 1, 28, 28).to(device)

# model = snn().to(device)
# output = model(random_data1, random_data2)
# print(output)
