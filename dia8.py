import streamlit as st
from datetime import time, datetime

st.header('st.slider')

# Ejemplo 1

st.subheader('Slider')

age2 = st.slider('How old are you?', 0, 130, 25)
st.write('Yo soy ', age2,'edad')

st.subheader("Prueba edad español")
age = st.slider('¿Cuántos años tienes?', 0, 130, 25)
st.write("I'm ", age, 'years old')

# Ejemplo 2

st.subheader('Range slider')

values = st.slider(
     'Select a range of values',
     0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

st.subheader("Rango deslizante")
valores =  st.slider(
    'Eligan un rango de valores',
    0.0, 100.0, (40.0, 60.0))
st.write("Valores: ", valores)

# Ejemplo 3

st.subheader('Range time slider')

appointment = st.slider(
     "Schedule your appointment:",
     value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)

# Ejemplo 4

st.subheader('Datetime slider')

start_time = st.slider(
     "When do you start?",
     value=datetime(2020, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)

st.subheader('Tiempo deslizante')
tiempo_empezado = st.slider(
    "Cuándo empiezas",
    value = datetime(2020,1,14,1,1),
    format = "MM/DD/YY - hh:mm")
st.write("Tiempo de inicio:", tiempo_empezado)

st.subheader("Rango de valores")
tiempo_sub = st.slider(
    "¿Cuántos meses quieres pronosticar?",
    0.0, 40.0, (20.0, 30.0)
)
st.write("Meses de pronóstico",tiempo_sub)