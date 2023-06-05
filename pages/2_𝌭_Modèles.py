import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid


excel_path = Path("data/Tables/Models.xlsx")
excel = pd.read_excel(excel_path, sheet_name="Models", header=0, keep_default_na=False, decimal=',')

st.dataframe(excel)
#st.write(excel.to_html(), unsafe_allow_html=True)