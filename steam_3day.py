
import streamlit as st
st.header('st.button')

if st.button('Say hello'):
     st.write("Adios")
else:
     st.write("Hola")