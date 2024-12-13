import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Using pickle for model persistence

# Load pre-trained model (replace with your model path)
model_path = "mymodel.joblib"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Function for data preprocessing
def preprocess_data(data):
    categorical_cols = [col for col in data.columns if data[col].dtype == object]
    numerical_cols = [col for col in data.columns if col not in categorical_cols]

    # Handle missing values (consider imputation techniques if needed)
    data.fillna(data.mean(), inplace=True)  # Replace with a more robust method

    # Label encoding for categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Feature scaling for numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data

# Function for prediction
def predict(data):
    preprocessed_data = preprocess_data(data.copy())
    features = preprocessed_data.drop("Loan_Status", axis=1)
    prediction = model.predict(features)[0]
    return prediction

# Streamlit App Layout
st.title("Loan Approval Prediction")

# User input section
st.subheader("Enter Applicant Information")
user_data = {}
for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Applicant_Income", "Coapplicant_Income", "Loan_Amount", "Loan_Term"]:
    user_data[col] = st.number_input(col)

# Convert categorical features to numerical (replace with dropdown menus if preferred)
user_data["Gender"] = 0 if user_data["Gender"] == "Male" else 1
user_data["Married"] = 0 if user_data["Married"] == "No" else 1
user_data["Education"] = 0 if user_data["Education"] == "Graduate" else 1
user_data["Self_Employed"] = 0 if user_data["Self_Employed"] == "No" else 1

# Prediction button
if st.button("Predict Loan Approval"):
    prediction = predict(pd.DataFrame(user_data, index=[0]))
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.warning("Loan Not Approved.")

# Display total data information (assuming you have pre-loaded total_data)
st.subheader("Total Data Information")
if "total_data" in st.session_state:  # Check if data is loaded in the session
    st.write(st.session_state["total_data"].describe(include="all"))

# Descriptive statistics for numerical variables
if "total_data" in st.session_state:
    describe_numericals(st.session_state["total_data"], [col for col in st.session_state["total_data"].columns if col not in ["Loan_Status", "Property_Area", "Loan_ID"]])

# Correlation matrix (consider using interactive libraries like plotly)
if "total_data" in st.session_state:
    plot_correlation_matrix(st.session_state["total_data"], [col for col in st.session_state["total_data"].columns if col not in ["Loan_Status", "Property_Area", "Loan_ID"]])

# Load total data for further analysis (optional)
upload_data = st.file_uploader("Upload Data (CSV)", type=["csv"])
if upload_data is not None:
    total_data = pd.read_csv(upload_data)
    st.session_state["total_data"] = total_data

# Function for descriptive statistics of numerical variables (optional for reusability)
def describe_numericals(data, numerical_cols):
    st.subheader("Descriptive Statistics (Numerical Variables)")
    st.write(data[numerical_cols].describe(include='all'))