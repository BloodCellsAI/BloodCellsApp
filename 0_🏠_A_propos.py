import streamlit as st
from PIL import Image
import os
from importlib import reload

st.markdown("""
# BloodCellsApp : une application de visualisation et de classification de cellules sanguines

## Contexte
La leucémie est un type de cancer du sang qui se caractérise par une production anormale de cellules sanguines, généralement des leucocytes, dans la moelle osseuse. Il existe plusieurs types de leucémies, dont les quatre les plus fréquentes sont la leucémie aiguë lymphoblastique (LAL), la leucémie aiguë myéloblastique (LAM), la leucémie lymphoïde chronique (LLC) et la leucémie myéloïde chronique (LMC). Les analyses de l'hémogramme permettent d'identifier des indices pouvant faire soupçonner la présence d'une leucémie.

L’émergence de l’intelligence artificielle dans le domaine de la médecine apporte de nouveaux outils d’analyse facilitant grandement les diagnostiques.
 """)

images_folder_path = r'data/Images'
image_path = os.path.join(images_folder_path, 'intro.jpg')
image = Image.open(image_path)
#st.image(image, use_column_width=True)


st.markdown("""
    ## Datasets

    #### Objectif

    Développer des modèles de deep-learning caractérisant différents types de leucocytes ainsi que de la segmentation sémantique permettant de recueillir des informations supplémentaires sur les cellules d’interets (taille, nombre …)

    #### 1. PBC dataset

    A publicly available [dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) of more than 
    17000 images of blood leukocytes from blood smears of healthy donors stained with MGG.<sup>[1](#footnote1)</sup>
    
    #### 2. APL dataset

    A publicly available [dataset](https://doi.org/10.6084/m9.figshare.14294675) of Images of peripheral blood smear results from 106 AML and APL patients.<sup>[2](#footnote2)</sup>

    #### 3. Munich dataset
    A publicly available [dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958). contains 18,365 expert-labeled single-cell images taken from peripheral blood smears of 100 patients diagnosed with Acute Myeloid Leukemia at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of hematological malignancy.<sup>[3](#footnote3)</sup>    

    #### 4. Raabin dataset
    A publicly available........ JR ?????
    

    Cell type| Cell Code|PBC|APL|Munich|
    |---------|----|---------|------|------|
    |neutrophils (segmented)| SNE|X|X|X|
    |eosinophils|              EO|X|X|X|
    |basophils|                BA|X|X|X|
    |lymphocytes|              LY|X|X|X|
    |monocytes|                MO|X|X|X|
    |metamyelocytes|          MMY|X|X|X|
    |myelocytes|               MY|X|X|X|
    |promyelocytes|           PMY|X|X|X|
    |band neutrophils|        BNE|X|X|X|
    |platelets|               PLT|X| | |
    |erythroblasts|           ERB|X|X|X|
    |smudges cell|           SMG| |X| |

    </br>

    <a name="footnote1">1.</a> *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. [Data Brief. 2020 Jun; 30: 105474.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)*

    <a name="footnote2">2.</a> *Deep learning for diagnosis of acute promyelocytic leukemia via recognition of genomically imprinted morphologic features. [npj Precis. Onc. 5, 38 (2021)](https://www.nature.com/articles/s41698-021-00179-y#citeas)*
 
    <a name="footnote3">3.</a> *Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks. [Nat Mach Intell 1, 538–544 (2019)](https://www.nature.com/articles/s42256-019-0101-9#citeas)*

    """, unsafe_allow_html=True)