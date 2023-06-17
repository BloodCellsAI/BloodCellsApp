# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:42:50 2023

@author: jrph1
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image,ImageDraw
import numpy as np
import torch.nn.functional as F
import os
# import streamlit as st

# html_temp = """
#     <div style="background-color:tomato;padding:10px">
#     <h1 style="color:white;text-align:center;font-family:'Times New Roman'; font-size: 56px; text-decoration: underline; font-weight: bold;"> Raabin </h1>
#     </div>
#     """

# st.markdown(html_temp, unsafe_allow_html=True)

# Continuer le reste du code ici
class MedianNormalizeTransform(object):
    def __call__(self, img):

       
        
        return img

image_transforms = transforms.Compose([
                       transforms.Resize((360, 360)),
                       transforms.ToTensor(),
                       MedianNormalizeTransform(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                   ])



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

model_pbc = DeeplabV3Plus(num_classes=12)
model_pbc.load_state_dict(torch.load('data/Models/PBC.pth', map_location=torch.device('cpu')))


reseau = torch.load('data/Models/Raabin.pth', map_location=torch.device('cpu'))
reseau.eval()

import streamlit as st

# def raabin():
#     html_temp = """
#         <div style="background-color:tomato;padding:10px">
#         <h1 style="color:white;text-align:center;font-family:'Times New Roman'; font-size: 56px; text-decoration: underline; font-weight: bold;"> Raabin </h1>
#         </div>
#         """
#     st.markdown(html_temp, unsafe_allow_html=True)

#     # Ajouter ici le reste du contenu pour la page Raabin

# def pbc():
#     html_temp = """
#         <div style="background-color:blue;padding:10px">
#         <h1 style="color:white;text-align:center;font-family:'Times New Roman'; font-size: 56px; text-decoration: underline; font-weight: bold;"> PBC </h1>
#         </div>
#         """
#     st.markdown(html_temp, unsafe_allow_html=True)
#     # Ajouter ici le reste du contenu pour la page PBC


# # Définir les pages
# pages = {
#     "Raabin": raabin,
#     "PBC": pbc,
# }

# # Utiliser la barre latérale pour sélectionner la page
# page = st.sidebar.selectbox("Choisissez une page", tuple(pages.keys()))

# # Appeler la fonction de la page sélectionnée
# pages[page]()

colors = np.array([(0, 0, 0),  # Arrière-plan
                       (255, 0, 0),  # Basophile
                       (0, 255, 0),  # Eosinophile
                       (255, 255, 0),  #Lymphocyte
                       (0,0, 255),  # Monocyte
                       (0, 255, 255)])  # Neutrophile

cell_names = ["Background",
              "Basophile",
              "Eosinophile",
              "Lymphocyte",
              "Monocyte",
              "Neutrophile"]    

def raabin():
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h1 style="color:white;text-align:center;font-family:'Times New Roman'; font-size: 56px; text-decoration: underline; font-weight: bold;"> Raabin </h1>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)


    # Ajouter ici le reste du contenu pour la page Raabin

    # Ajouter un explorateur de fichiers pour charger une image
    uploaded_file = st.file_uploader("Choisissez une image (JPEG/PNG)", type=["jpg", "png"])
    if uploaded_file is not None:
        # Charger l'image
        image = Image.open(uploaded_file)
        
        img = image_transforms(image)

        with torch.no_grad():
            output = reseau(img.unsqueeze(0))
            
        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        colored_image = colors[output]
        colored_image = Image.fromarray(np.uint8(colored_image))

        # Afficher l'image chargée à gauche
        #st.image(image, caption='Image chargée', use_column_width=True)
        legend_image = Image.new("RGB", (200, image.height), (255, 255, 255))
        legend_draw = ImageDraw.Draw(legend_image)
        cell_height = image.height // len(cell_names)
        for i, name in enumerate(cell_names):
            color = tuple(colors[i])
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            legend_draw.rectangle([(0, y_start), (50, y_end)], fill=color)
            legend_draw.text((60, y_start), name, fill=(0, 0, 0))
        col1, col2,col3 = st.columns(3)
        col1.image(image, caption='Image chargée', use_column_width=True)
        col2.image(colored_image, caption='Image inférée', use_column_width=True)
        col3.image(legend_image, caption='Légende', use_column_width=True)


def pbc():
    html_temp = """
        <div style="background-color:blue;padding:10px">
        <h1 style="color:white;text-align:center;font-family:'Times New Roman'; font-size: 56px; text-decoration: underline; font-weight: bold;"> PBC </h1>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    # Ajouter ici le reste du contenu pour la page PBC
    colors = np.array([(0, 0, 0),  # Arrière-plan
                       (255, 0, 0),  # Basophile
                       (0, 255, 0),  # BNE
                       (0, 0, 255),  # SNE
                       (255, 255, 0),  # Eosinophile
                       (255, 0, 255),  # Érythroblaste
                       (0, 255, 255),  # MY
                       (255, 128, 0),  # MMY
                       (255, 0, 128),  # PMY
                       (128, 255, 0),  # Lymphocyte
                       (255, 0, 128),  # Monocyte
                       (128, 0, 255)])  # Plaquette

    cell_names = ["Arrière-plan",
                  "Basophile",
                  "BNE",
                  "SNE",
                  "Eosinophile",
                  "Érythroblaste",
                  "MY",
                  "MMY",
                  "PMY",
                  "Lymphocyte",
                  "Monocyte",
                  "Plaquette"]

    # Ajouter un explorateur de fichiers pour charger une image
    uploaded_file = st.file_uploader("Choisissez une image (JPEG/PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        # Charger l'image
        image = Image.open(uploaded_file)
        
        img = image_transforms(image)

        with torch.no_grad():
            output = model_pbc(img.unsqueeze(0))
            
        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        colored_image = colors[output]
        colored_image = Image.fromarray(np.uint8(colored_image))

        # Afficher l'image chargée à gauche
        #st.image(image, caption='Image chargée', use_column_width=True)
        legend_image = Image.new("RGB", (200, image.height), (255, 255, 255))
        legend_draw = ImageDraw.Draw(legend_image)
        cell_height = image.height // len(cell_names)
        for i, name in enumerate(cell_names):
            color = tuple(colors[i])
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            legend_draw.rectangle([(0, y_start), (50, y_end)], fill=color)
            legend_draw.text((60, y_start), name, fill=(0, 0, 0))
        col1, col2,col3 = st.columns(3)
        col1.image(image, caption='Image chargée', use_column_width=True)
        col2.image(colored_image, caption='Image inférée', use_column_width=True)
        col3.image(legend_image, caption='Légende', use_column_width=True)


# Définir les pages
pages = {
    "Raabin": raabin,
    "PBC": pbc,
}

# Utiliser la barre latérale pour sélectionner la page
page = st.sidebar.selectbox("Choisissez une page", tuple(pages.keys()))

# Appeler la fonction de la page sélectionnée
pages[page]()