import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_data = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/imagefoder/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)

test_data = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/imagefoder/test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)

valid_data = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/imagefoder/valid', transform=transform)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=False, num_workers=4)

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 50
train_loss_list, valid_loss_list, train_acc_list, valid_acc_list = [], [], [], []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total
    train_loss = train_loss / train_total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)

    # Validation loss
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            valid_correct += (predicted == labels).sum().item()
            valid_total += labels.size(0)

    valid_accuracy = 100 * valid_correct / valid_total
    valid_loss = valid_loss / valid_total
    valid_loss_list.append(valid_loss)
    valid_acc_list.append(valid_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
test_loss = test_loss / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label="Train")
plt.plot(valid_loss_list, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label="Train")
plt.plot(valid_acc_list, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

cm = confusion_matrix(all_labels, all_preds)
classes = ('Acetaldehyde', 'Acetone', 'Ammonia', 'Benzene', 'Butanol', 'CO', 'Ethylene', 'Methane', 'Methanol', 'Toluene')
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
