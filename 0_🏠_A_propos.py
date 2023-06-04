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
#st.image(image, use_column_width=True)

st.text(' La leucémie est un type de cancer du sang se caractérisant par une production anormale\n de cellule sanguine nommées leucocyte. L’émergence de l’intelligence artificielle dans le\n domaine de la médecine apport de nouveaux outils d’analyse facilitant grandement les\n diagnostiques.')

st.markdown("""
    ## Datasets

    #### 1. PBC dataset

    A publicly available [dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) of more than 
    17000 images of blood leukocytes from blood smears of healthy donors stained with MGG.<sup>[1](#footnote1)</sup>
    
    #### 2. APL dataset

    A publicly available [dataset](https://doi.org/10.6084/m9.figshare.14294675) of Images of peripheral blood smear results from 106 AML and APL patients.<sup>[2](#footnote2)</sup>

    #### 3. Munich dataset
    A publicly available [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958). contains 18,365 expert-labeled single-cell images taken from peripheral blood smears of 100 patients diagnosed with Acute Myeloid Leukemia at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of hematological malignancy.<sup>[3](#footnote3)</sup>    

    

      |Cell type| Cell Code|PBC|APL|Munich|
    |---------|----|---------|------|------|
    |neutrophils (segmented)| SNE|X| |X|
    |eosinophils|              EO|X|X|X|
    |basophils|                BA|X|X|X|
    |lymphocytes|              LY|X|X|X|
    |monocytes|                MO|X|X|X|
    |metamyelocytes|          MMY|X| |X|
    |myelocytes|               MY|X| |X|
    |promyelocytes|           PMY|X| |X|
    |band neutrophils|        BNE|X| |X|
    |platelets|               PLT|X| | |
    |erythroblasts|           ERB|X| |X|

    </br>
    </br>

    <a name="footnote1">1.</a> *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)*

    <a name="footnote2">2.</a> *Deep learning for diagnosis of acute promyelocytic leukemia via recognition of genomically imprinted morphologic features. [npj Precis. Onc. 5, 38 (2021)](https://www.nature.com/articles/s41698-021-00179-y#citeas)*
 
    <a name="footnote3">3.</a> *Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. [Nat Mach Intell 1, 538–544 (2019)](https://www.nature.com/articles/s42256-019-0101-9#citeas)*

    """, unsafe_allow_html=True)