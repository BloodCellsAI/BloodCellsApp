import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

excel_path = Path("data/Tables/Models.xlsx")
df = pd.read_excel(excel_path, sheet_name="Models", header=0, keep_default_na=False, decimal=',')

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
    height=350, 
    width='100%',
    reload_data=True
)

selected = grid_response['selected_rows'][0]

#selected_row = AgGrid(df, gridOptions=gridOptions, height=300)

st.text(selected)

Modèle = selected['Modèle']
Mask = selected['Mask']
Augmentation = selected['Augmentation']
Pretrain = selected['Pretrain']
Fine_tuning = selected['Fine-tunning']
Train = selected['Train']

Filename = str(Modèle + '_' + Mask + '_' + Augmentation + '_')

st.text(Filename)









#AgGrid(excel)


#st.dataframe(excel)
#st.write(excel.to_html(), unsafe_allow_html=True)

#selection = st.table(excel)


