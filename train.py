import torch
import torch.nn as nn
import os
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from network import snn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

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
    lr = 0.002
    print(f"Learning rate: {lr}")
    batch_size = 64  # Increased batch size

    training_data = dataset(csv_file='sign_data/train_data.csv', root_dir='sign_data/Dataset/train/', transform=transform)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_data = dataset(csv_file='sign_data/test_data.csv', root_dir='sign_data/Dataset/test/', transform=transform)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = snn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    start_time = time.time()
    
    best_accuracy = 0

    loss_list = []
    acc_list = []


    for epoch in range(epochs):
        print('-'*90)
        print(f"Epoch {epoch+1}:")
        model.train()
        total_loss = 0
        for i, (img1, img2, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            with autocast():
                output = model(img1, img2)
                loss = criterion(output.squeeze(), label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

        print('TOTAL LOSS:', total_loss, 'LEN:', len(train_loader))
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        acc = validate(model, criterion, val_loader)
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        loss_list.append(avg_loss)
        acc_list.append(acc)
        
        scheduler.step()
            
    end_time = time.time()
    torch.save(model.state_dict(), 'model_last.pth')
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    plot_metrics(loss_list, acc_list)


def validate(model, criterion, val_loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader, desc='Validation'):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = criterion(output.squeeze(), label)
            total_loss += loss.item()

            # Compute accuracy
            pred = torch.sigmoid(output).round()
            total_correct += pred.eq(label.view_as(pred)).sum().item()
            total_samples += label.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    return accuracy

def test_all():
    val_data = dataset(csv_file='sign_data/test_data.csv', root_dir='sign_data/Dataset/test/', transform=transform)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = snn().to(device)
    model.load_state_dict(torch.load('model.pth'))
    criterion = nn.BCEWithLogitsLoss()
    
    acc = validate(model, criterion, val_loader)
    print(f"Accuracy: {acc:.4f}")

def plot_metrics(loss, acc):
    epochs = range(1, len(loss) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, loss, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.plot(epochs, acc, label='Validation Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy over Epochs')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.savefig('metrics.png')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'test':
            test_all()
    else:
        print("Usage: python train.py [train|test]")