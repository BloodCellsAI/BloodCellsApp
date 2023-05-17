

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DeeplabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeeplabV3Plus, self).__init__()
        
        

        # Encoder
        self.resnet = models.resnet18(pretrained=True)
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.layer1 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1) # 64
        self.layer2 = self.resnet.layer2 # 128
        self.layer3 = self.resnet.layer3 # 256
        self.layer4 = self.resnet.layer4 # 512

        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.aspp2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(256 * 5, 256, kernel_size=1, stride=1)

        # Decoder
        self.conv3 = nn.Conv2d(256, 48, kernel_size=1, stride=1)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(320, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

    # ASPP
        aspp1 = self.aspp1(x4)
        aspp2 = self.aspp2(x4)
        aspp3 = self.aspp3(x4)
        aspp4 = self.aspp4(x4)
        global_avg_pool = self.global_avg_pool(x4)
        global_avg_pool = self.conv1(global_avg_pool)
        global_avg_pool = F.interpolate(global_avg_pool, size=aspp4.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate
        concat = torch.cat((aspp1, aspp2, aspp3, aspp4, global_avg_pool), dim=1)
        concat = self.conv2(concat)

        # Decoder
        dec0 = self.conv3(concat)
        dec1 = self.up1(dec0)
        dec1 = F.interpolate(dec1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec1, x3), dim=1)
        dec3 = self.conv4(dec2)
        dec4 = self.conv5(dec3)
        dec5 = self.up2(dec4)
        dec5 = F.interpolate(dec5, size=x1.size()[2:], mode='bilinear', align_corners=True) # Fix size mismatch
        dec6 = torch.cat((dec5, x1), dim=1)
        dec7 = self.conv6(dec6)

        return F.interpolate(dec7, size=x.size()[2:], mode='bilinear', align_corners=True)

model = DeeplabV3Plus(num_classes=9)
if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')
# Boucle d'entraînement
# Boucle d'entraînement
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.callbacks import TensorBoard
import glob
from PIL import Image
import numpy as np
# Créer le scheduler
#scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

# Paramètres d'entraînement
num_epochs = 205
batch_size = 15
learning_rate = 0.01
momentum=0.9
weight_decay=0.000001

# Put the model on the CPU
device = torch.device('cuda')
model.to(device)

tensorboard = TensorBoard(log_dir='./logs')
# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# Transformation des données
data_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.1,hue=0.1,contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Chargement des données
image_filenames = glob.glob('C:/Bloodcells/Modele/Transpose/Image/*.png')
label_filenames = glob.glob('C:/Bloodcells/Modele/Transpose/Label/*.png')



class TransformLabels:
    def __call__(self, label):
        label[label == 0] = 0
        label[label == 100] = 1
        label[label == 110] = 2
        label[label == 120] = 3
        label[label == 130] = 4
        label[label == 140] = 5
        label[label == 150] = 6
        label[label == 160] = 7
        label[label == 170] = 8

       
        return label.to(torch.long)


for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    angles=np.random.random_integers(1,360,1)
    dataset = [(data_transforms(Image.open(image_fn).rotate(angles)), torch.from_numpy(np.array(Image.open(label_fn).rotate(angles)))) for image_fn, label_fn in zip(image_filenames, label_filenames)]
    dataset = [(data, TransformLabels()(label)) for (data, label) in dataset]

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Décrémentation du learning rate
    if (epoch+1) % 250 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
        print(f"Learning rate updated: {optimizer.param_groups[0]['lr']:.6f}")
    
    for i, (images, labels) in enumerate(train_loader):
        # Envoi des données sur le device
        images, labels = images.to(device), labels.long().to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # Calcul de l'accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.numel()
        correct += (predicted == labels).sum().item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader)
    accuracy = 100 * correct / total
    writer.add_scalar('loss', loss.item(), epoch)
    writer.add_scalar('accuracy',accuracy,epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Accuracy: {accuracy:.2f}%")
torch.save(model.state_dict(), 'C:/Bloodcells/Modele/Transpose/Pytorch2104_median.pth')
writer.close()

