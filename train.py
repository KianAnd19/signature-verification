import torch
import torch.nn as nn
import os
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from network import snn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = kagglehub.dataset_download("robinreni/signature-verification-dataset")

class dataset(Dataset):
    def __init__(self, pairs_data, root_dir, transform=None):
        self.pairs_frame = pairs_data.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs_frame)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 0])
        img2_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 1])
        img1 = Image.open(img1_name).convert("L")
        img2 = Image.open(img2_name).convert("L")
        label = float(self.pairs_frame.iloc[idx, 2])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def create_simple_split():
    """Use only training data and create our own clean train/test split"""
    # Load only the training data - ignore the problematic test split entirely
    all_data = pd.read_csv(f'{path}/sign_data/train_data.csv', header=None, names=['img1', 'img2', 'label'])
    
    print(f"Total samples: {len(all_data)}")
    print(f"Genuine pairs (label=0): {(all_data['label'] == 0).sum()}")
    print(f"Forged pairs (label=1): {(all_data['label'] == 1).sum()}")
    
    # Create stratified split to maintain class balance
    train_data, test_data = train_test_split(
        all_data, 
        test_size=0.2, 
        stratify=all_data['label'], 
        random_state=42
    )
    
    print(f"\nAfter clean split:")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, test_data

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Larger size for better performance
    transforms.ToTensor()
])

def train():
    epochs = 30 
    lr = 1e-4 
    print(f"Learning rate: {lr}")
    batch_size = 64 

    # Create simple clean split
    train_df, test_df = create_simple_split()
    
    # Both datasets use the same root directory (train folder)
    training_data = dataset(pairs_data=train_df, root_dir=f'{path}/sign_data/train/', transform=transform)
    val_data = dataset(pairs_data=test_df, root_dir=f'{path}/sign_data/train/', transform=transform)

    # Dataloaders
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model = snn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
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

            output = model(img1, img2)
            loss = criterion(output.squeeze(), label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        acc = validate(model, criterion, val_loader)
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        loss_list.append(avg_loss)
        acc_list.append(acc)
        
    end_time = time.time()
    torch.save(model.state_dict(), 'model_last.pth')
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")
    
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
    # Use the same simple split for testing
    _, test_df = create_simple_split()
    val_data = dataset(pairs_data=test_df, root_dir=f'{path}/sign_data/train/', transform=transform)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    model = snn().to(device)
    model.load_state_dict(torch.load('model_last.pth', weights_only=True))
    criterion = nn.BCEWithLogitsLoss()
    
    acc = validate(model, criterion, val_loader)
    print(f"Final Test Accuracy: {acc:.4f}")

def image_similarity(img1_path, img2_path):
    model = snn().to(device)
    model.load_state_dict(torch.load('model_last.pth', weights_only=True))

    # Preprocess images
    img1 = transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        logit = model(img1, img2)
        similarity = torch.sigmoid(logit).item()

    return similarity

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
    if len(sys.argv) < 2:
        print("Usage: python train.py [train | val | test img1 img2]")
        sys.exit(1)
        
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'val':
        test_all()
    elif sys.argv[1] == 'test' and len(sys.argv) == 4:
        similarity = image_similarity(sys.argv[2], sys.argv[3])
        print('Genuine' if similarity < 0.5 else 'Forged')
    else:
        print("Usage: python train.py [train | val | test img1 img2]")
