import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar el modelo
model_path = "mymodel.joblib"
model = joblib.load(model_path)

# Cargar datos
data_path1 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/train.csv"
data_path2 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/test.csv"
train_data = pd.read_csv(data_path1)
test_data = pd.read_csv(data_path2)
total_data = pd.concat([train_data, test_data], ignore_index=True)

# Imputación de valores faltantes y codificación
categorical = [var for var in total_data.columns if total_data[var].dtype == 'object']
numerical = [var for var in total_data.columns if total_data[var].dtype != 'object']

imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer(strategy='median')

total_data.loc[:, categorical] = imputer_cat.fit_transform(total_data[categorical])
total_data.loc[:, numerical] = imputer_num.fit_transform(total_data[numerical])

total_data["Gender"] = total_data["Gender"].apply(lambda x: 1 if x == "Male" else 0)
total_data["Self_Employed"] = total_data["Self_Employed"].apply(lambda x: 1 if x == "Yes" else 0)
total_data["Loan_Status"] = total_data["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)
total_data["Education"] = total_data["Education"].apply(lambda x: 1 if x == "Graduate" else 0)
total_data["Married"] = total_data["Married"].apply(lambda x: 1 if x == "Yes" else 0)
total_data["Dependents"] = total_data["Dependents"].replace("3+", "3")

categorical.remove('Loan_ID')

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
num_plots = len(categorical)
num_rows = (num_plots + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))

for idx, cat_col in enumerate(categorical):
    row, col = idx // 2, idx % 2
    sns.countplot(x=cat_col, data=total_data, hue='Loan_Status', ax=axes[row, col])
    axes[row, col].set_title(cat_col)

if num_plots % 2 != 0:
    fig.delaxes(axes.flatten()[-1])

plt.subplots_adjust(hspace=0.5)
st.pyplot(fig)

# Gráficos de variables numéricas
st.header("Gráficos de Variables Numéricas")
num_rows = (len(numerical) + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 6))

for i, var in enumerate(numerical):
    row = i // 2
    col = i % 2
    sns.histplot(data=total_data, x=var, kde=True, color='skyblue', bins=20, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var}')
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Density')

if num_rows * 2 > len(numerical):
    fig.delaxes(axes.flatten()[-1])

plt.tight_layout()
st.pyplot(fig)

# Distribución de la variable de salida
st.header("Distribución de la Variable de Salida")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=total_data, x='Loan_Status', ax=ax)
ax.set_title('Distribución de la Variable de Salida')
st.pyplot(fig)

# Matriz de correlación
st.header("Matriz de Correlación")
corr_matrix = total_data[numerical].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Preparación de los datos para el modelo
scaler = StandardScaler()
total_data[numerical] = scaler.fit_transform(total_data[numerical])

X = total_data.drop(columns=['Loan_Status', 'Property_Area', 'Loan_ID'])
y = total_data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento y predicción del modelo
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Mostrar el valor del accuracy del model.py
st.header("Valor del Accuracy")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy del modelo optimizado: {accuracy:.2f}")  # Ajustar según los resultados en model.py

# Gráficos de la matriz de dispersión y curva ROC
st.header("Matriz de Dispersión y Curva ROC")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['No', 'Yes'], columns=['Pred No', 'Pred Yes'])
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Matriz de Dispersión")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Verdadero")

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
axes[1].plot(fpr, tpr, label=f'ROC Curve (area = {auc_score:.2f})')
axes[1].plot([0, 1], [0, 1], 'r--', label='No Skill')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Curva ROC')
axes[1].legend()

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


# Botón para generar la predicción
if st.button("Generar Predicción"):
    # Realizar la predicción
    prediction = model.predict(input_data)[0]
    
    # Mostrar el resultado de la predicción
    st.write("Predicción del Estado del Préstamo:", "Aprobado" if prediction == 1 else "Rechazado")


# Sección de Referencias Bibliográficas
st.header("Referencias Bibliográficas")
st.write("""
- Dataset de Kaggle: [Finance Loan Approval Prediction Data](https://www.kaggle.com/krishnaraj30/finance-loan-approval-prediction-data)
- Documentación de Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Documentación de Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)
""")
