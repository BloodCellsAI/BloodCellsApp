from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st
import random

###
# Chargement des données
###

datasets = ["PBC", "APL", "Munich"]
dataset = st.sidebar.selectbox("Choose a dataset", datasets)

if dataset == "PBC":
    image = Image.open("data/Images/PBC_8classes.jpg")
elif dataset == 'APL':
    image = Image.open("data/Images/APL_dataset.jpg")
elif dataset =="Munich":
    image = Image.open("data/Images/Munich_dataset.jpg")
st.image(image, use_column_width=True)

st.markdown("Image source[^1]")

if dataset == "PBC":
    df = pd.read_csv('data/Tables/PBC.csv')
elif dataset == "APL":
    df = pd.read_csv('data/Tables/APL.csv')
elif dataset =="Munich":
    df = pd.read_csv('data/Tables/Munich.csv')

# elimination du file path et Photo_id
df = df.drop(['Photo_path',"Photo_id","Patient_ID"], axis=1, errors='ignore')

###
#Data sélection
###
continu = df.select_dtypes('number').columns
discrete = df.select_dtypes('object').columns
#varaibles_selected = st.selectbox("Choose variables :", variables)
col1, col2 = st.columns(2)

###
#Display graphs
###

with col1: 
    discrete_selected = st.selectbox("Choisir une variable discrète", discrete)

    fig = plt.figure()
    sns.countplot(y=discrete_selected, data=df)
    st.pyplot(fig)

with col2:
    continu_selected = st.selectbox("Choisur une variable continue :", continu)

    fig = plt.figure()
    sns.histplot(data=df, x=continu_selected)
    st.pyplot(fig)

"""
#### Echantillon d'images
"""

def sample_display(dataset, figsize=(20, 6)):
    dataset_path = Path('data/Sample/' + str(dataset))

    Cell_types = [d.parts[-1] for d in dataset_path.glob('*') if d.is_dir()]

    fig, axs = plt.subplots(2, len(Cell_types), figsize=figsize, sharey='row')

    for j, cell_type in enumerate(Cell_types):
        if dataset in ['PBC']:
            image = random.choice([i.parts[-1] for i in Path(dataset_path/cell_type).glob('*.jpg') if i.is_file()])
        if dataset in ['APL', 'Munich']:
            image = random.choice([i.parts[-1] for i in Path(dataset_path/cell_type).glob('*.png') if i.is_file()])
        image_path = Path(dataset_path/cell_type/image)

        img = plt.imread(image_path)
        axs[0][j].imshow(img)
        axs[0][j].set_title(cell_type)

        

        colors = ['red', 'green', 'blue']
        for i, col in enumerate(colors):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            axs[1][j].plot(histr, color = col)

    
    return fig

if st.button("Générer"):
    st.pyplot(sample_display(dataset=dataset))
#else:
    #st.pyplot(sample_display(dataset=dataset))



st.markdown("""
[^1]Author : By Original: A. Rad Vector: RexxS, Mikael Häggström and birdy and Mikael Häggström, M.D. Author info- Reusing images- Conflicts of interest:NoneMikael Häggström, M.D. - Own work based on: Hematopoiesis (human) diagram.svg, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=9420824
""")
