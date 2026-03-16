import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("Machine Learning Auto Trainer")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])


# ---------------- FILE VALIDATION ---------------- #

if uploaded_file is None:
    st.warning("Please upload a CSV dataset to continue")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("Uploaded dataset is empty")
        st.stop()

except Exception:
    st.error("Invalid CSV file")
    st.stop()


# ---------------- DATA PREVIEW ---------------- #

st.subheader("Dataset Preview")
st.dataframe(df)

st.subheader("Missing Values")
st.write(df.isnull().sum())


# ---------------- TARGET COLUMN ---------------- #

target = st.selectbox("Select Target Column", df.columns)

X = df.drop(target, axis=1)
y = df[target]


# ---------------- ENCODING ---------------- #

le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = le.fit_transform(X[col])

if y.dtype == "object":
    y = le.fit_transform(y)


# ---------------- HANDLE MISSING VALUES ---------------- #

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# ---------------- SPLIT DATA ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------- ALGORITHM SELECTION ---------------- #

algo = st.selectbox(
    "Choose Algorithm",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN",
        "K-Means Clustering",
        "Linear Regression"
    ],
)


# ---------------- TRAIN MODEL ---------------- #

if st.button("Train Model"):

    if algo == "Logistic Regression":
        model = LogisticRegression()

    elif algo == "Decision Tree":
        model = DecisionTreeClassifier()

    elif algo == "Random Forest":
        model = RandomForestClassifier()

    elif algo == "SVM":
        model = SVC()

    elif algo == "KNN":
        model = KNeighborsClassifier()
    
    elif algo == "K-Means Clustring":
        model  = KMeans()

    elif algo == "Linear Regression":
        model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    st.success(f"Model R2 Score: {score:.2f}")


    # ---------------- PREDICTION TABLE ---------------- #

    st.subheader("Prediction Results")

    result = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions
        }
    )

    st.dataframe(result)


    # ---------------- VISUALIZATION ---------------- #

    st.subheader("Prediction Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x=predictions)
    st.pyplot(fig)


    st.subheader("Feature Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(12,8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    st.pyplot(fig2)