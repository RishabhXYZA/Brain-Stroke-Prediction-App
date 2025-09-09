import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set up page configuration
st.set_page_config(
    page_title="Brain Stroke Prediction App",
    page_icon="🧠",
    layout="wide",
)

# Title and introduction
st.title("🧠 Brain Stroke Prediction App")
st.write("This application predicts the likelihood of a brain stroke based on various health indicators.")
st.write("Please provide the patient's information in the sidebar.")

# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# --- Data Preprocessing and Model Training ---

# Drop 'id' and handle 'Other' gender
df = df.drop('id', axis=1)
df = df[df['gender'] != 'Other']

# Add an age group column
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 30, 50, 100],
    labels=['Young', 'Middle', 'Older']
)

# Separate features (X) and target (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- Define Preprocessing Pipeline ---

# Define features
numerical_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = [
    'gender', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'smoking_status', 'age_group'
]

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- Apply Preprocessing and SMOTE ---

# First, apply the preprocessor to transform the data
X_processed = preprocessor.fit_transform(X)

# Now, use SMOTE on the preprocessed numerical data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)

# --- Train the final model on the resampled data ---

# Choose the best model
best_model = LinearDiscriminantAnalysis()
best_model.fit(X_resampled, y_resampled)

# --- Sidebar for user input ---
st.sidebar.header("Patient Data Input")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", X['gender'].unique())
    age = st.sidebar.slider("Age", min_value=0, max_value=100, value=50, step=1)
    hypertension = st.sidebar.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
    ever_married = st.sidebar.selectbox("Ever Married", X['ever_married'].unique())
    work_type = st.sidebar.selectbox("Work Type", X['work_type'].unique())
    Residence_type = st.sidebar.selectbox("Residence Type", X['Residence_type'].unique())
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", min_value=50.0, max_value=300.0, value=90.0)
    bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=80.0, value=25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", X['smoking_status'].unique())


    # Create age_group for input
    if age < 30:
        age_group = 'Young'
    elif age < 50:
        age_group = 'Middle'
    else:
        age_group = 'Older'

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'age_group': age_group
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display user input and prediction ---
st.subheader("Patient Information Provided")
st.dataframe(input_df)

# Apply preprocessing to the input data before prediction
input_processed = preprocessor.transform(input_df)
prediction_proba = best_model.predict_proba(input_processed)
prediction = np.argmax(prediction_proba)

st.subheader("Prediction Result")
if prediction == 1:
    st.error("The model predicts a *high risk of stroke*.Please Consult to the doctor.")
    st.write(f"Confidence score: {prediction_proba[0][1]*100:.2f}%")
else:
    st.success("The model predicts a *low risk of stroke*.Otherwise,you feel the symptoms,consult to the doctor.")
    st.write(f"Confidence score: {prediction_proba[0][0]*100:.2f}%")

# --- Visualizations ---
st.markdown("---")
st.header("Exploratory Data Analysis (EDA) from the Original Dataset")

# Show distributions of numerical features
st.subheader("Distribution of Numerical Features")
fig_hist = px.histogram(df, x=numerical_features, color_discrete_sequence=px.colors.qualitative.Plotly)
st.plotly_chart(fig_hist, use_container_width=True)

# Stroke vs gender
st.subheader("Stroke Rate by Gender")
stroke_gender = df.groupby('gender')['stroke'].mean() * 100
fig_gender = px.bar(
    stroke_gender,
    x=stroke_gender.index,
    y=stroke_gender.values,
    title='Percentage of Stroke by Gender',
    labels={'x': 'Gender', 'y': 'Stroke Percentage (%)'},
    color=stroke_gender.index
)
st.plotly_chart(fig_gender, use_container_width=True)

# Stroke vs hypertension
st.subheader("Stroke Rate by Hypertension Status")
stroke_hypertension = df.groupby('hypertension')['stroke'].mean() * 100
fig_hypertension = px.bar(
    stroke_hypertension,
    x=stroke_hypertension.index,
    y=stroke_hypertension.values,
    title='Percentage of Stroke by Hypertension Status',
    labels={'x': 'Hypertension (0=No, 1=Yes)', 'y': 'Stroke Percentage (%)'},
    color=stroke_hypertension.index
)
st.plotly_chart(fig_hypertension, use_container_width=True)

