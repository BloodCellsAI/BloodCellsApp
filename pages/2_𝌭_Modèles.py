import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Load data
excel_path = Path("data/Tables/Models.xlsx")
df = pd.read_excel(excel_path, sheet_name="Models", header=0, keep_default_na=False, decimal=',')


# Define AgGrid options
gb = GridOptionsBuilder.from_dataframe(df)
#gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
#gb.configure_side_bar() #Add a sidebar
gb.configure_selection('single', use_checkbox=True) #Enable row selection
gridOptions = gb.build()

grid_response = AgGrid(
    df,
    gridOptions=gridOptions,
    data_return_mode='AS_INPUT', 
    update_mode='MODEL_CHANGED', 
    fit_columns_on_grid_load=True,
    enable_enterprise_modules=False,
    height=450, 
    width='100%',
    reload_data=True
)

# Selected boxes
selected = grid_response['selected_rows'][0]

Modèle = selected['Modèle']
Mask = selected['Mask']
Augmentation = selected['Augmentation']
Pretrain = selected['Pretrain']
Fine_tuning = selected['Fine-tunning']
Train = selected['Train']


###
# Display images
###
if (Modèle == 'VGG16' and Mask == 'Y' and Augmentation == 'Y' and Pretrain=="Imagenet" and Fine_tuning=="4_Layers" and Train=="PBC_6cat") : 
    st.image("data/Images/VGG16_6cat_matrice.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/Images/VGG16_6cat_lr.jpg")
    with col2:
        st.image("data/Images/VGG16_6cat_resume.jpg")

if (Modèle == 'VGG16' and Mask == 'Y' and Augmentation == 'Y' and Pretrain=="Imagenet" and Fine_tuning=="N" and Train=="PBC_6cat") : 
    pass

if (Modèle == 'VGG16' and Mask == 'Y' and Augmentation == 'Y' and Pretrain=="Imagenet" and Fine_tuning=="N" and Train=="PBC_8cat") : 
    st.image("data/Images/VGG16_8cat_matrice.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/Images/VGG16_8cat_lr.jpg")
    with col2:
        st.image("data/Images/VGG16_8cat_resume.jpg")

if (Modèle == 'VGG16' and Mask == 'Y' and Augmentation == 'Y' and Pretrain=="Imagenet" and Fine_tuning=="4_Layers" and Train=="PBC_11cat") : 
    st.image("data/Images/VGG16_11cat_matrice.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image("data/Images/VGG16_11cat_lr.jpg")
    with col2:
        st.image("data/Images/VGG16_11cat_resume.jpg")

