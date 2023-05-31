import streamlit as st
from PIL import Image
import os
from importlib import reload

st.title("BloodCellsApp : une application de visualisation et de classification de cellules sanguines")

st.markdown("""
## Contexte
La leucémie est un type de cancer du sang se caractérisant par une production anormale\n de cellule sanguine nommées leucocyte. L’émergence de l’intelligence artificielle dans le\n domaine de la médecine apport de nouveaux outils d’analyse facilitant grandement les\n diagnostiques.
 """)

st.title('Bloodcell Prediction version 352.4')
st.header('Nicolas ORIEUX, Jean-Robert PHILIPPE & Florian FREYTET')

images_folder_path = r'data/Images'
image_path = os.path.join(images_folder_path, 'intro.jpg')
image = Image.open(image_path)
st.image(image, use_column_width=True)

st.text(' La leucémie est un type de cancer du sang se caractérisant par une production anormale\n de cellule sanguine nommées leucocyte. L’émergence de l’intelligence artificielle dans le\n domaine de la médecine apport de nouveaux outils d’analyse facilitant grandement les\n diagnostiques.')