from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st

###
# Chargement des données
###

datasets = ["PBC", "APL"]
dataset = st.sidebar.selectbox("Choose a dataset", datasets)


base_image = Image.open("data/Images/Hematopoiesis.png")
st.image(base_image)
"""
Author : By Original: A. Rad Vector: RexxS, Mikael Häggström and birdy and Mikael Häggström, M.D. Author info- Reusing images- Conflicts of interest:NoneMikael Häggström, M.D. - Own work based on: Hematopoiesis (human) diagram.svg, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=9420824
"""

if dataset == "PBC":
    boxes = {'boxes' : [[0,0,100,100],[10,20,50,150]]}

if dataset == "PBC":
    df = pd.read_csv('data/Tables/PBC.csv')
elif dataset == "APL":
    df = pd.read_csv('data/Tables/APL.csv')

###
# Hematopoïesis image
###





###
#Data sélection
###




###
# Display graphs
###

# basic graph
fig = plt.figure()
sns.countplot(y="Cell_type", data=df)

st.pyplot(fig)