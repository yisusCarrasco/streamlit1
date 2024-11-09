import streamlit as st 
import pandas as pd
import numpy as np 
import altair as alt

st.title("Practica")

st.markdown('<h1>Empezamos a practicar en el día <span style="color:red;">5</span></h1>', unsafe_allow_html=True)

st.write(1234)

df = pd.DataFrame({
    "primera columna":[1,2,3,4],
    "segunda columna":[10,20,30,40]}
)

st.write("Publicamos el dataframe",df,"Después de publicar el dataset")

df2 = pd.DataFrame(
    np.random.randn(200,3),
    columns = ["a","b","c"]
)
c = alt.Chart(df2).mark_circle().encode(
    x="a", y = "b", size="c", tooltip = ["a","b","c"]
)
st.write(c)



