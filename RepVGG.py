import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root='imagefolder/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='imagefolder/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

validset = torchvision.datasets.ImageFolder(root='imagefolder/valid', transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False)

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if stride == 1:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = None
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.conv(x))
        else:
            if self.stride == 1:
                identity = self.shortcut(x)
            else:
                identity = 0
            return self.relu(self.bn1(self.conv1(x)) + self.bn2(self.conv2(x)) + identity)


    def _fuse_bn(self, conv, bn):
        w_conv = conv.weight
        b_conv = conv.bias

        if b_conv is None:
            b_conv = torch.zeros(w_conv.size(0), device=w_conv.device)

        mean_bn, var_bn, w_bn, b_bn = bn.running_mean, bn.running_var, bn.weight, bn.bias
        w_fused = w_conv * (w_bn.reshape(-1, 1, 1, 1) / torch.sqrt(var_bn.reshape(-1, 1, 1, 1) + 1e-5))
        b_fused = (b_conv - mean_bn) * (w_bn / torch.sqrt(var_bn + 1e-5)) + b_bn

    # Return a new Conv2D layer
        return nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True
        ).to(conv.weight.device).requires_grad_().to(conv.weight.device)

    def switch_to_deploy(self):
        fused_conv1 = self._fuse_bn(self.conv1, self.bn1)
        fused_conv2 = self._fuse_bn(self.conv2, self.bn2)

        fused_conv1.weight.data = fused_conv1.weight.data + fused_conv2.weight.data
        fused_conv1.bias.data = fused_conv1.bias.data + fused_conv2.bias.data

        if self.stride == 1 and self.shortcut is not None and isinstance(self.shortcut, nn.Conv2d):
            fused_conv1.weight.data += self.shortcut.weight.data
        self.conv = fused_conv1

    # Remove other layers for deployment
        for name, module in self.named_children():
            if name not in ['conv', 'relu']:
                self.add_module(name, None)
        self.deploy = True

class RepVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(RepVGG, self).__init__()

        self.stage_0 = RepVGGBlock(3, 64, kernel_size=3, stride=2, padding=1)
        self.stage_1 = nn.Sequential(
            RepVGGBlock(64, 64, kernel_size=3, stride=1, padding=1),
            RepVGGBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.stage_2 = nn.Sequential(
            RepVGGBlock(64, 128, kernel_size=3, stride=2, padding=1),
            RepVGGBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.stage_3 = nn.Sequential(
            RepVGGBlock(128, 256, kernel_size=3, stride=2, padding=1),
            RepVGGBlock(256, 256, kernel_size=3, stride=1, padding=1),
            RepVGGBlock(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.stage_4 = nn.Sequential(
            RepVGGBlock(256, 512, kernel_size=3, stride=2, padding=1),
            RepVGGBlock(512, 512, kernel_size=3, stride=1, padding=1),
            RepVGGBlock(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stage_0(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RepVGG(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 50
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    net.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    valid_loss = running_loss / len(validloader)
    valid_accuracy = correct / total
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')

net.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print(cm)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(valid_losses, label='Valid Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(valid_accuracies, label='Valid Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()
