import torch
import torch.nn as nn
import os
import pandas as pd
from network import snn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
        label = float(self.pairs_frame.iloc[idx, 2])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

def train():
    # hyperparameters
    epochs = 10
    lr = 1e-3
    batch_size = 64  # Increased batch size

    training_data = dataset(csv_file='sign_data/train_data.csv', root_dir='sign_data/Dataset/train/', transform=transform)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = snn().to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    start_time = time.time()
    
    print(model.cnn.conv1.weight.grad)


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            with autocast():
                output = model(img1, img2)
                loss = criterion(output.squeeze(), label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    end_time = time.time()
    
    print(model.cnn.conv1.weight.grad)
    torch.save(model.state_dict(), 'model.pth')
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")

def validate():
    val_data = dataset(csv_file='sign_data/test_data.csv', root_dir='sign_data/Dataset/test/', transform=transform)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = snn().to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    
    with torch.no_grad():
        total_accuracy = 0
        for i, (img1, img2, label) in enumerate(val_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = criterion(output.squeeze(), label)
            total_loss += loss.item()

            # compute accuracy
            pred = torch.sigmoid(output).round()
            correct = pred.eq(label.view_as(pred)).sum().item()
            accuracy = correct / len(label)
            total_accuracy += accuracy
            
        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        total_accuracy /= len(val_loader)
        print(f"Validation Accuracy: {total_accuracy:.4f}")
if __name__ == "__main__":
    # train()
    validate()