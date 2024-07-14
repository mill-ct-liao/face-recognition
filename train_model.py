import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# Define the custom dataset class
class AgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for age in range(20, 51):  # Ages from 20 to 50
            age_dir = os.path.join(root_dir, str(age))
            if os.path.isdir(age_dir):
                for img_name in os.listdir(age_dir):
                    if img_name.endswith('.jpg') or img_name.endswith('.png'):
                        self.image_paths.append(os.path.join(age_dir, img_name))
                        self.labels.append(age - 20)  # Age labels from 0 to 30

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the model
class AgeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AgeClassifier, self).__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), self.num_classes)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_wts = model.state_dict()
    best_auc = 0.0

    # Check the number of parameters
    num_params = count_parameters(model)
    if num_params > 1e6:
        print(f"Warning: The model has {num_params} parameters, which exceeds 1 million and may not be suitable for deployment on edge devices.")
    else:
        print(f"The model has {num_params} parameters which is less than 1 million.")
    # Lists to store metrics
    train_loss_history = []
    val_loss_history = []
    train_auc_history = []
    val_auc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_labels = []
            all_probs = []

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.softmax(dim=1).detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')

            # Save metrics
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_auc_history.append(epoch_auc)
            else:
                val_loss_history.append(epoch_loss)
                val_auc_history.append(epoch_auc)

            print(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = model.state_dict()

    print(f'Best val AUC: {best_auc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Visualize metrics
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Training Loss')
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_auc_history, label='Training AUC')
    plt.plot(epochs, val_auc_history, label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

def main():    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    dataset = AgeDataset("/home/mill/Desktop/face-recognition/resampled_dataset", transform=transform)
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    num_classes = 31  # Ages from 20 to 50 mapped to 0 to 30
    model = AgeClassifier(num_classes)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model
    model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25)
    
    # Save the trained model
    torch.save(model.state_dict(), 'age_classifier.pth')
    print("Model saved as 'age_classifier.pth'")
    
if __name__ == "__main__":
    main()