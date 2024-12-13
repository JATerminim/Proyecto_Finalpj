import pandas as pd
import matplotlib.pyplot as plt


# Download latest version
data_path1 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/train.csv"  # Replace with the correct file path
train_data = pd.read_csv(data_path1)
# train_data.head()

data_path2 = "/home/vscode/.cache/kagglehub/datasets/krishnaraj30/finance-loan-approval-prediction-data/versions/1/test.csv"  # Replace with the correct file path
test_data = pd.read_csv(data_path2)

# Suponiendo que train_data y test_data son DataFrames de Pandas
total_data = pd.concat([train_data, test_data], ignore_index=True)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pickle

# EDA (ejemplo)
#total_data.describe()

# EDA Detallado

# 1. Exploración inicial
#print(total_data.head())
#print(total_data.info())
#print(total_data.describe())

#print(total_data.shape, "\n") 
#print(total_data.columns, "\n") 

categorical = [var for var in total_data.columns if total_data[var].dtype == 'object']
numerical = [var for var in total_data.columns if total_data[var].dtype != 'object']

#print("Categorical Features:", categorical)
#print("Numerical Features:", numerical)
#print()

for var in numerical:
    print(f"{var}: [{total_data[var].min()}, {total_data[var].max()}]")

# 5. Identificación de valores faltantes
#print(total_data.isnull().sum())

# 6. Identificación de outliers

# método1
# Eliminar filas donde 'Credit_History' tiene valores faltantes
#total_data = total_data.dropna(subset=['Credit_History'])

#total_data.isna().sum()

# 6. Identificación de outliers
 from sklearn.impute import SimpleImputer

# Impute missing values in categorical features with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
total_data.loc[:, categorical] = categorical_imputer.fit_transform(total_data[categorical])

# total_data.isna().sum()

# method3

# Impute missing values in numerical features with median
numerical_imputer = SimpleImputer(strategy='median')
total_data.loc[:, numerical] = numerical_imputer.fit_transform(total_data[numerical])

# total_data.isna().sum()

# Remove 'Loan_ID' from the list of categorical variables
categorical.remove('Loan_ID')


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming categorical is a list of categorical variable names
num_plots = len(categorical)

# Generate subplots dynamically
num_rows = (num_plots + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 5 * num_rows))

# Iterate over categorical variables
for idx, cat_col in enumerate(categorical):
    row, col = idx // 2, idx % 2
    # Create count plot for current categorical variable
    sns.countplot(x=cat_col, data=total_data, hue='Loan_Status', ax=axes[row, col])
    axes[row, col].set_title(cat_col)

# Remove empty subplots if any
if num_plots % 2 != 0:
    fig.delaxes(axes.flatten()[-1])

# Adjust subplot spacing
plt.subplots_adjust(hspace=0.5)

# Show plot
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the number of rows needed
num_rows = (len(numerical) + 1) // 2

# Plot histograms for all numerical features
fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 6))

for i, var in enumerate(numerical):
    row = i // 2
    col = i % 2
    sns.histplot(data=total_data, x=var, kde=True, color='skyblue', bins=20, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var}')
    axes[row, col].set_xlabel(var)
    axes[row, col].set_ylabel('Density')

# Remove empty subplots if any
if num_plots % 2 != 0:
    fig.delaxes(axes.flatten()[-1])

# Adjust layout
plt.tight_layout()
plt.show()

# Count the occurrences of each category in the output variable
output_var_counts = total_data['Loan_Status'].value_counts()

# Plot the counts using Seaborn's countplot
plt.figure(figsize=(6, 4))
sns.countplot(data=total_data, x='Loan_Status')
plt.title('Distribution of Output Variable')
plt.xlabel('Output Variable')
plt.ylabel('Count')
plt.show()


# Calculate correlation matrix using only numerical columns
corr_matrix = total_data[numerical].corr()

# Generate heatmap
sns.heatmap(corr_matrix, annot=True)
plt.show()

# codificacion

total_data["Gender"] = total_data["Gender"].apply(lambda x: 1 if x == "Male" else 0)
total_data["Self_Employed"] = total_data["Self_Employed"].apply(lambda x: 1 if x == "Yes" else 0)
total_data["Loan_Status"] = total_data["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)
total_data["Education"] = total_data["Education"].apply(lambda x: 1 if x == "Graduate" else 0)
total_data["Married"] = total_data["Married"].apply(lambda x: 1 if x == "Yes" else 0)
total_data["Dependents"] = total_data["Dependents"].replace("3+", "3")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

total_data[numerical] = scaler.fit_transform(total_data[numerical])

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

X = total_data.drop(columns=['Loan_Status','Property_Area','Loan_ID'], axis = 1)
y = total_data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# X_train

from sklearn.linear_model import LogisticRegression


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


regresion_cm = confusion_matrix(y_test, y_pred)

# Dibujaremos esta matriz para hacerla más visual
cm_df = pd.DataFrame(regresion_cm)

plt.figure(figsize = (3, 3))
sns.heatmap(cm_df, annot=True, fmt="d", cbar=False)

plt.tight_layout()

plt.show()


# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

# Plot ROC Curve
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'r--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression Model')
plt.legend()
plt.grid()
plt.show()

from sklearn.model_selection import GridSearchCV

hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid = GridSearchCV(model, hyperparams, scoring = "accuracy", cv = 10)
grid

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")

model_grid = LogisticRegression(penalty = "l1", C = 10, solver = "liblinear")
model_grid.fit(X_train, y_train)
y_pred = model_grid.predict(X_test)

grid_accuracy = accuracy_score(y_test, y_pred)
grid_accuracy

import joblib

try:
    joblib.dump(model_grid,"mymodel.joblib")
except Exception as e:
    print(f"Error: {e}")



