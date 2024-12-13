import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Título de la aplicación
st.title("Predicción de Aprobación de Préstamos")

# Cargar el modelo pre-entrenado
try:
    model = joblib.load("mymodel.joblib")  # Asegúrate de que la ruta sea correcta
except FileNotFoundError:
    st.error("No se encontró el modelo entrenado. Verifica la ruta del archivo 'mymodel.joblib'.")
    exit()

# Utilizar el DataFrame existente (asumiendo que 'total_data' ya está definido)
# ... (Asegúrate de que 'total_data' contenga los datos correctos y las columnas esperadas)

# Separar las características (X) y la variable objetivo (y)
try:
    X = total_data.drop(columns=["Loan_Status", "Property_Area", "Loan_ID"], axis=1)
except KeyError:
    st.error("Error: Alguna de las columnas especificadas no existe en el DataFrame.")
    st.write("Columnas disponibles:", total_data.columns)
    exit()

y = total_data["Loan_Status"]

# Crear un formulario para ingresar los datos del usuario
st.subheader("Ingresa los datos del solicitante")

# Crear campos de entrada para cada característica (ajusta los nombres y tipos según tus datos)
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, min_value=X[col].min(), max_value=X[col].max())

# Escalar los datos de entrada del usuario
scaler = StandardScaler()
user_input_scaled = scaler.fit_transform(pd.DataFrame([user_input]).values)

# Realizar la predicción
prediction = model.predict(user_input_scaled)[0]
probability = model.predict_proba(user_input_scaled).max() * 100

# Mostrar el resultado de la predicción
if prediction == 1:
    result = "Préstamo aprobado"
else:
    result = "Préstamo rechazado"

st.success(f"Resultado: {result}")
st.write(f"Probabilidad: {probability:.2f}%")

# Agregar información adicional (opcional)
st.write("**Nota:** Esta predicción se basa en un modelo de machine learning y puede no ser 100% precisa. Consulta con un experto financiero para una evaluación más completa.")