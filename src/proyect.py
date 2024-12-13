import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Cargar el modelo
model_path = "mymodel.joblib"
model = joblib.load(model_path)

# Cargar datos
data_path1 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/train.csv"
data_path2 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/test.csv"
train_data = pd.read_csv(data_path1)
test_data = pd.read_csv(data_path2)
total_data = pd.concat([train_data, test_data], ignore_index=True)

# Variables categóricas y numéricas
categorical = [var for var in total_data.columns if total_data[var].dtype == 'object']
numerical = [var for var in total_data.columns if total_data[var].dtype != 'object']

# Convertir variables categóricas a numéricas
total_data["Gender"] = total_data["Gender"].apply(lambda x: 1 if x == "Male" else 0)
total_data["Self_Employed"] = total_data["Self_Employed"].apply(lambda x: 1 if x == "Yes" else 0)
total_data["Loan_Status"] = total_data["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)
total_data["Education"] = total_data["Education"].apply(lambda x: 1 if x == "Graduate" else 0)
total_data["Married"] = total_data["Married"].apply(lambda x: 1 if x == "Yes" else 0)
total_data["Dependents"] = total_data["Dependents"].replace("3+", "3")

# Imputación de valores faltantes
imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer(strategy='median')
total_data[categorical] = imputer_cat.fit_transform(total_data[categorical])
total_data[numerical] = imputer_num.fit_transform(total_data[numerical])

# Dividir los datos para entrenamiento y prueba
X = total_data.drop(columns=['Loan_Status','Property_Area','Loan_ID'], axis=1)
y = total_data["Loan_Status"]

# Normalización de las características numéricas
scaler = StandardScaler()
X[numerical] = scaler.fit_transform(X[numerical])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Asegurarse de que y_test y y_pred sean del mismo tipo
y_pred = model.predict(X_test)
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)
grid_accuracy = accuracy_score(y_test, y_pred)

# Configuración de la página
st.set_page_config(page_title="Predicción de Aprobación de Préstamos", layout="wide")

# Título de la aplicación
st.title("Predicción de Aprobación de Préstamos")
st.write("Visualización y predicción usando un modelo de regresión logística optimizado")

# Sección de visualización del dataset
st.header("Visualización del Dataset")
st.write(total_data)

# Estadísticos descriptivos
st.header("Estadísticos Descriptivos")
st.write(total_data.describe())

# Gráficos de variables categóricas
st.header("Gráficos de Variables Categóricas")
fig, axes = plt.subplots(len(categorical), 1, figsize=(10, len(categorical) * 3))
for i, column in enumerate(categorical):
    sns.countplot(data=total_data, x=column, ax=axes[i])
st.pyplot(fig)

# Gráficos de variables numéricas
st.header("Gráficos de Variables Numéricas")
fig, axes = plt.subplots(len(numerical), 1, figsize=(10, len(numerical) * 3))
for i, column in enumerate(numerical):
    sns.histplot(data=total_data, x=column, kde=True, ax=axes[i])
st.pyplot(fig)

# Identificación de valores faltantes
st.header("Valores Faltantes")
st.write(total_data.isnull().sum())

# Identificación de outliers
st.header("Outliers")
fig, axes = plt.subplots(len(numerical), 1, figsize=(10, len(numerical) * 3))
for i, column in enumerate(numerical):
    sns.boxplot(data=total_data, x=column, ax=axes[i])
st.pyplot(fig)

# Matriz de correlación
st.header("Matriz de Correlación")
corr_matrix = total_data[numerical].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Mostrar el modelo seleccionado
st.header("Modelo Seleccionado")
st.write("Modelo: Regresión Logística")

# Mostrar los resultados de la optimización de parámetros
st.header("Resultados de Optimización de Parámetros")
st.write("Mejores hiperparámetros: {'penalty': 'l1', 'C': 10, 'solver': 'liblinear'}")  # Ajustar según los resultados del grid search

# Mostrar el valor del accuracy
st.header("Valor del Accuracy")
st.write(f"Accuracy del modelo optimizado: {grid_accuracy}")

# Gráficos de la matriz de dispersión y curva ROC
st.header("Matriz de Dispersión y Curva ROC")
st.subheader("Matriz de Dispersión")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Matriz de Dispersión")
ax.set_xlabel("Predicción")
ax.set_ylabel("Verdadero")
st.pyplot(fig)

st.subheader("Curva ROC")
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (area = {auc_score:.2f})')
ax.plot([0, 1], [0, 1], 'r--', label='No Skill')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Curva ROC')
ax.legend()
st.pyplot(fig)

# Predicción con valores del usuario
st.header("Predicción")
st.write("Ingrese los valores para hacer una predicción:")
gender = st.selectbox("Género", ["Male", "Female"])
married = st.selectbox("Casado", ["Yes", "No"])
education = st.selectbox("Educación", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Autoempleado", ["Yes", "No"])
dependents = st.selectbox("Dependientes", ["0", "1", "2", "3+"])
loan_amount = st.number_input("Cantidad del Préstamo", min_value=0.0, step=0.1)
loan_amount_term = st.number_input("Plazo del Préstamo (en meses)", min_value=0)
credit_history = st.selectbox("Historial de Crédito", ["1.0", "0.0"])
applicant_income = st.number_input("Ingreso del Solicitante", min_value=0.0, step=0.1)
coapplicant_income = st.number_input("Ingreso del Co-Solicitante", min_value=0.0, step=0.1)

# Crear un DataFrame con las mismas columnas y en el mismo orden que X
input_data = pd.DataFrame({
    'Gender': [1 if gender == "Male" else 0],
    'Married': [1 if married == "Yes" else 0],
    'Education': [1 if education == "Graduate" else 0],
    'Self_Employed': [1 if self_employed == "Yes" else 0],
    'Dependents': [dependents.replace("3+", "3")],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [float(credit_history)]
})

# Normalización de las características numéricas del input
input_data[numerical] = scaler.transform(input_data[numerical])

# Asegurar el orden de las columnas coincida con el del modelo entrenado
input_data = input_data[X.columns]

# Realizar la predicción
prediction = model.predict(input_data)[0]

# Mostrar el resultado de la predicción
st.write("Predicción del Estado del Préstamo:", "Aprobado" if prediction == 1 else "Rechazado")

# Sección de Referencias Bibliográficas
st.header("Referencias Bibliográficas") 
st.write(""" 
         - Dataset de Kaggle: [Finance Loan Approval Prediction Data](https://www.kaggle.com/krishnaraj30/finance-loan-approval-prediction-data) 
         - Documentación de Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) 
         - Documentación de Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/) """)