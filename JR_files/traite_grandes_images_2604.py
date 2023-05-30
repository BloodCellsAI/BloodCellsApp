
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import torch.nn as nn

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Chargement du modèle
#reseau = torch.load('C:/Bloodcells/Modele/Data/Raabin/GrTh/Learn/pytorch0604_median.pth', map_location=torch.device('cpu'))
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

reseau = DeeplabV3Plus(num_classes=9)
reseau.load_state_dict(torch.load('C:/Bloodcells/Modele/Transpose/Pytorch2104_median.pth'))

reseau.eval()

# Transformation de l'image
image_transforms = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ])

# Définition des couleurs pour chaque classe
colors = np.array([(0, 0, 0),  # Background
                   (128, 0, 0),  # Classe 1
                   (0, 128, 0),  # Classe 2
                   (128, 128, 0),  # Classe 3
                   (0, 0, 128),  # Classe 4
                   (128, 0, 128),  # Classe 5
                   (0, 128, 128),  # Classe 6
                   (128, 128, 128),  # Classe 7
                   (64, 0, 0)])  # Classe 8

# Fonction pour traiter une carrelette
def process_tile(tile, model):
    with torch.no_grad():
        output = model(tile.unsqueeze(0))
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    colored_image = colors[output]    
    colored_image = Image.fromarray(np.uint8(colored_image))
    return colored_image

from PIL import ImageFilter

def process_image(img_path, tile_size, overlap, model):
    img = Image.open(img_path)
    img_size = img.size
    
    # Redimensionner l'image
    new_size = tuple(int(x*2.0) for x in img_size)
    img = img.resize(new_size, resample=Image.BICUBIC)
    img_size = new_size
    
    tiles = []
    for i in range(0, new_size[1] - tile_size + 1, tile_size - overlap):
        for j in range(0, new_size[0] - tile_size + 1, tile_size - overlap):
            tile = img.crop((j, i, j + tile_size, i + tile_size))
            tiles.append(tile)
            
    tiles_processed = []
    for tile in tiles:
        tile = image_transforms(tile)
        colored_tile = process_tile(tile, model)
        tiles_processed.append(colored_tile)
        
    # Interpoler entre les carreaux
    for i in range(1, len(tiles_processed)):
        tiles_processed[i-1] = tiles_processed[i-1].filter(ImageFilter.SMOOTH)
        tiles_processed[i] = tiles_processed[i].filter(ImageFilter.SMOOTH)
        
    output_size = new_size
    output = Image.new('RGB', output_size)
    index = 0
    for i in range(0, output_size[1] - tile_size + 1, tile_size - overlap):
        for j in range(0, output_size[0] - tile_size + 1, tile_size - overlap):
            tile_overlap = tiles_processed[index].crop((0, 0, tile_size, overlap))
            output_overlap = output.crop((j, i, j + tile_size, i + overlap))
            overlap_mask = np.maximum(np.array(tile_overlap), np.array(output_overlap))
            output.paste(Image.fromarray(overlap_mask), (j, i))
            output.paste(tiles_processed[index].crop((0, overlap, tile_size, tile_size)), (j, i + overlap))
            index += 1
            
    # Redimensionner l'image de sortie à sa taille d'origine
    #output = output.resize(img_size, resample=Image.BICUBIC)
    
    return output


# Chemin des images à traiter
img_paths = glob.glob("C:/Bloodcells/Modele/Data/Patients/All/All/Patient_49/Unsigned slides/*.jpg")


# Taille des carrelettes et chevauchement entre deux carrelettes
tile_size = 360
overlap = 180
from matplotlib.patches import Patch

# Définition des étiquettes de légende
labels = ['Background', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 'Classe 5', 'Classe 6', 'Classe 7', 'Classe 8']





# Boucle sur les images
for img_path in img_paths:
    print(f"Processing image {img_path}")
    output = process_image(img_path, tile_size, overlap, reseau)
    # Récupérer le nom du fichier et remplacer l'extension .bmp par _label.bmp
    label_path = os.path.splitext(img_path)[0] + '_label.png'
    # Enregistrer l'image de Label après une rotation de 270 degrés
    output.save(label_path)
    colored_image=output
    # Création d'une liste de patches avec les couleurs correspondantes
    patches = [Patch(color=c/255, label=labels[i]) for i, c in enumerate(colors)]
    # Configuration de l'affichage du graphique
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_image, alpha=0.8) # Définir une transparence pour contrôler la luminosité de l'image
    plt.axis('off') # Désactiver les axes
    plt.grid(False) # Désactiver la grille
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='black', labelcolor='white') # Ajouter la légende avec fond noir et caractères blancs
    plt.tight_layout() # Réduire les marges autour du graphique
    plt.show()
    plt.savefig(label_path, dpi=300, bbox_inches='tight', pad_inches=0) # Enregistrer l'image modifiée avec la légende           