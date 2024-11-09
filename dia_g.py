import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

st.title("Predicción de exportación de uvas frescas usando GRU")

# Cargar y procesar el conjunto de datos
data_path = 'https://raw.githubusercontent.com/yisusCarrasco/data/main/UTP%20Cantidades%20de%20uva%20exportada%20-%20Hoja%201.csv'
df = pd.read_csv(data_path, index_col='Fecha', parse_dates=True)
df.index.freq = 'MS'
df["Cantidad(tn)"] = df["Cantidad(tn)"].str.replace(",", ".").astype(float)
df["Precio($) Prom"] = df["Precio($) Prom"].str.replace(",", ".").astype(float)
df = df.drop('Cantidad(kg)', axis=1)

# Mostrar datos iniciales
st.subheader("Datos iniciales")
st.write(df.head())

# Matriz de correlación
st.subheader("Matriz de correlación")
correlacion = df.corr(method='pearson', min_periods=10)
fig, ax = plt.subplots()
sns.heatmap(correlacion, cmap='coolwarm', annot=True, ax=ax)
st.pyplot(fig)

# Graficar columnas
st.subheader("Cantidad (tn), Precio ($) Prom, y FOB")
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
df["Cantidad(tn)"].plot(ax=ax[0], title="Cantidad (tn)")
df["Precio($) Prom"].plot(ax=ax[1], title="Precio($) Prom")
df["FOB"].plot(ax=ax[2], title="FOB")
st.pyplot(fig)

# Preparar datos para el modelo
df_final = df.resample('M').mean()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_final)

sequence_length = 10
num_features = len(df_final.columns)

sequences, labels = [], []
for i in range(len(scaled_data) - sequence_length):
    sequences.append(scaled_data[i:i + sequence_length])
    labels.append(scaled_data[i + sequence_length][0])

sequences, labels = np.array(sequences), np.array(labels)
train_size = int(0.8 * len(sequences))
train_x, test_x = sequences[:train_size], sequences[train_size:]
train_y, test_y = labels[:train_size], labels[train_size:]

# Crear el modelo
model = Sequential()
model.add(LSTM(32, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.005), loss='mean_squared_error')

# Entrenar el modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
history = model.fit(train_x, train_y, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Mostrar la pérdida
st.subheader("Pérdida del Modelo")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_title("Model Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

# Evaluación del modelo
model = tf.keras.models.load_model('best_model.h5')
test_loss = model.evaluate(test_x, test_y)
st.write("Pérdida en el conjunto de prueba:", test_loss)

# Predecir
predictions = model.predict(test_x)
mae = np.mean(np.abs(predictions - test_y))
mse = np.mean((predictions - test_y) ** 2)
rmse = np.sqrt(mse)

st.write("MAE:", mae)
st.write("MSE:", mse)
st.write("RMSE:", rmse)

# Graficar predicciones y valores reales
fig, ax = plt.subplots()
ax.plot(test_y, label='Real')
ax.plot(predictions, label='Predicción')
ax.set_title("Predicción vs Real")
ax.legend()
st.pyplot(fig)
