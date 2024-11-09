import streamlit as st
import pandas as pd
import numpy as np

st.header("Selección selectbox")

option = st.selectbox(
    "Cuál es tu preferido color",
    ("","Azul", "rojo", "verde")
)

st.write("Tu color favorito es:",option)

