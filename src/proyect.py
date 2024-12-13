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
#data_path1 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/train.csv"
#data_path2 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/test.csv"
train_data = pd.read_csv("../src/train_data0")
test_data = pd.read_csv("../src/test_data0")
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

st.write("El dataset de aprobación de préstamos personales de Kaggle contiene información sobre los solicitantes de préstamos y sus historiales crediticios. Las variables del problema son:")

st.write(" + `Gender`: Género del solicitante (Masculino/Femenino).")

st.write(" + `Married`: Estado civil del solicitante (Casado/Soltero).")

st.write(" + `Dependents`: Número de dependientes del solicitante.")

st.write(" + `Education`: Nivel educativo del solicitante (Graduado/No Graduado).")

st.write(" + `Self_Employed`: Estado de empleo del solicitante (Empleado = 0 /Desempleado =  1).")

st.write(" + `ApplicantIncome`: Ingreso mensual del solicitante.")

st.write(" + `CoapplicantIncome`: Ingreso mensual del co-aplicante (si aplica).")

st.write(" + `LoanAmount`: Monto del préstamo solicitado.")

st.write(" + `Loan_Amount_Term`: Plazo del préstamo (en años).")

st.write(" + `Credit_History`: Historial crediticio del solicitante (1 si hay historial, 0 si no).")

st.write(" + `Property_Area`: Área residencial del solicitante (Urbana/Rural).")

st.write(" + `Loan_Status`: Estado del préstamo (Aprobado/Rechazado).")

st.write("")
st.write("")
st.write("Este dataset es crucial para desarrollar modelos predictivos que ayuden a las instituciones financieras a tomar decisiones informadas sobre la aprobación de préstamos. Al utilizar técnicas de regresión logística, se puede predecir la probabilidad de que un solicitante sea aprobado o rechazado basado en sus características y historial crediticio. Esto ayuda a minimizar el riesgo crediticio y mejorar la eficiencia en el proceso de aprobación de préstamos.")
st.write("")
st.write("")
#st.write("Visualización y predicción usando un modelo de regresión logística optimizado")

# st.write("Visualización y predicción usando un modelo de regresión logística optimizado")

# Sección de visualización del dataset
st.header("Visualización del Dataset")
st.write(total_data)

# Estadísticos descriptivos
st.header("Estadísticos Descriptivos")
st.write(total_data.describe())

# Gráficos de variables categóricas
st.header("Gráficos de Variables Categóricas")

st.write("A continuacion visualizamos los graficos de las variables del tipo categorico en relacion al estado del prestamo")
 
num_plots = len(categorical)
num_rows = (num_plots + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))

st.write("De aqui podemos apreciar que:")
st.write(" + La poblacion con mas creditos aprobados son del sexo masculino, por otro lado la cantidad de mujeres que poseen creditos aprobados sin similares a la cantidad de hombres con creditos rechazados.")
st.write(" + La cantidad de personas con estado civil casados son las que poseen mas aprobación de creditos respecto a las personas cuyo estado civil es soltero.")
st.write(" + Los clientes que no disponen carga familiar son las que poseen mas creditos aprobados.")
st.write(" + Los clientes con mas presencia de asignacion y aprobacion de creditos son los que poseen empleo.")
st.write(" + Los clientes que posee un grado de estudio son favorecidos al momento de asignar y aprobar creditos. ")
st.write(" + Área residencial del solicitante no pareciera influir al momento de asignar y aprobar los creditos.")
st.write(" + Se aprecia que existen mas clientes con creditos aprobado.")



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
st.write("De aqui podemos apreciar que:")
st.write(" + La distribucion las variables ingreso mensual del aplicante y co-aplicante es asimetrica a la derecha")
st.write(" + La distribucion del monto solicitado tiende a ser simetrica.")
st.write(" + Las distribuciones de las varabiables plazo del prestamo e historial del prestamo son dispersas ya que son conteos.")


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
# st.header("Distribución de la Variable de Salida")
# fig, ax = plt.subplots(figsize=(6, 4))
# sns.countplot(data=total_data, x='Loan_Status', ax=ax)
#ax.set_title('Distribución de la Variable de Salida')
# st.pyplot(fig)

# Matriz de correlación
st.header("Matriz de Correlación")
corr_matrix = total_data[numerical].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write(" + La matriz de correlacion revela que existe una relacion lineal positiva moderada del `55%` entre el aplicante y el monto solicitado, seguido por una relacion debil del `18%` entre el co-aplicante y el monto solitado")
st.write(" + no existe relacion fuerte entre las otras variables.")

st.header("Seleccion del modelo")
st.write(" La aprobación de préstamos es un problema binario, donde el resultado puede ser `Aprobado` o `Rechazado`. En este sentido, es adecuado considersr el modelo de regresión logística, dado que está diseñada específicamente para manejar problemas binarios, lo que la hace una elección natural para esta tarea.")


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
st.write("Una vez implementado entrenado y optimizado el modelo se tiene que:")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"+ el accuracy del modelo optimizado: {accuracy:.2f}, indicando que el modelo fue capaz de realizar el 80% de las predicciones correctas.")  # Ajustar según los resultados en model.py
st.write(" ")

# Gráficos de la matriz de dispersión y curva ROC
st.write("Seguidamente se aprecia la matriz de dispersión")
# fig, axes = plt.subplots(1, 1, figsize=(20, 8))

# Calcular la matriz de confusión 
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index=['No', 'Yes'], columns=['Pred No', 'Pred Yes']) 

# Graficar la matriz de confusión 
fig, ax = plt.subplots(figsize=(6, 4)) 
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax) 
ax.set_title("Matriz de Confusión") 
ax.set_xlabel("Predicción") 
ax.set_ylabel("Verdadero") 
plt.tight_layout() 
plt.show()

st.pyplot(fig)

st.write("La matriz de dispersion nos indica que:")
st.write(" + Verdaderos positivos: tenemos 11 caso en los que el modelo predijo no aprobación del credito y en efecto se confirma con la data que asi fue.")
st.write(" + Verdaderos negativos: tenemos 147 casos en lo que el modelo predijo aprobación del credito y en efecto se confirma con la data que asi fue.")
st.write(" + Falsos positivos: tenemos 8 casos en los que el modelo predijo no aprobación y en la data realmente fue aprobado.")
st.write(" + Falsos negativos: tenemos 31 casos en los que el modelo predijo aprobación y realmente no fue aprobado.")



# Predicción con valores del usuario
st.header("Predicción del Modelo")
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
- Credit scoring, aplicando técnicas de regresión logística y redes neuronales, para una cartera de microcrédito : [https://repositorio.uasb.edu.ec/bitstream/10644/6872/1/T2962-MGFARF-Montalvan-Credit.pdf](https://repositorio.uasb.edu.ec/bitstream/10644/6872/1/T2962-MGFARF-Montalvan-Credit.pdf)
- Regresión logística v/s Arboles de decisión en el riesgo crediticio: [https://revista.ccaitese.com/index.php/ridt/article/view/21/13](https://revista.ccaitese.com/index.php/ridt/article/view/21/13)
""")
