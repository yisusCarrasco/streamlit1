import pandas as pd
import numpy as np
import streamlit as st

nummbers = np.random.choice(range(1,10000),100000,replace=True)
st.header("Valores aleatorios y gráfico lineal")

df = pd.DataFrame(
    np.random.randn(30,3),
    columns=["a","b","c"]
)

st.line_chart(df)



