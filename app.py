import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("🔥 ML Trainer (Full Visual Version)")


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    st.subheader("📊 Original Dataset")
    st.dataframe(df)

    # TARGET
    target = df.columns[-1]
    input_cols = df.columns[:-1]

    st.success(f"🎯 Target: {target}")

    mode = st.radio("Mode", ["Supervised", "Unsupervised"])

    # ---------------- DATA TRANSFORMATION ---------------- #
    st.subheader("🔄 Dataset Transformation")

    if mode == "Supervised":
        st.write("✅ Target column USED")
        st.dataframe(df)
    else:
        st.write("❌ Target column REMOVED")
        st.dataframe(df.drop(columns=[target]))

    # ---------------- ENCODING VISUAL ---------------- #
    st.subheader("🔍 Encoding Visualization")

    X_original = df[input_cols].copy()
    X_encoded = df[input_cols].copy()

    encoders = {}

    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            encoders[col] = le

    col1, col2 = st.columns(2)

    with col1:
        st.write("🟢 Before Encoding")
        st.dataframe(X_original.head())

    with col2:
        st.write("🔵 After Encoding")
        st.dataframe(X_encoded.head())

    # ---------------- ENCODING MAP ---------------- #
    st.subheader("📘 Encoding Mapping")

    for col, le in encoders.items():
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.write(f"{col} → {mapping}")

    # ---------------- PREPROCESS ---------------- #
    X = X_encoded.copy()
    X = SimpleImputer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    # PCA
    if st.checkbox("Use PCA"):
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    else:
        pca = None

    # ---------------- SUPERVISED ---------------- #
    if mode == "Supervised":

        y = df[target]

        if y.dtype == "object":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        else:
            target_encoder = None

        algo = st.selectbox("Algorithm", [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "KNN", 
            "Linear Regression"
        ])

        if st.button("🚀 Train"):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if algo == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algo == "Decision Tree":
                model = DecisionTreeClassifier()
            elif algo == "Random Forest":
                model = RandomForestClassifier()
            elif algo == "SVM":
                model = SVC()
            elif algo == "Linear Regression":
                model = LinearRegression()
            else:
                model = KNeighborsClassifier()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"🎯 Accuracy: {acc:.2f}")

            st.session_state.update({
                "model": model,
                "mode": mode,
                "encoders": encoders,
                "input_cols": input_cols,
                "target_encoder": target_encoder,
                "pca": pca
            })

    # ---------------- UNSUPERVISED ---------------- #
    else:

        algo = st.selectbox("Clustering Algorithm", [
            "K-Means", "DBSCAN", "Hierarchical"
        ])

        if algo == "K-Means":
            k = st.slider("Clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)
            clusters = model.fit_predict(X)

        elif algo == "DBSCAN":
            eps = st.slider("EPS", 0.1, 5.0, 0.5)
            model = DBSCAN(eps=eps)
            clusters = model.fit_predict(X)

        else:
            k = st.slider("Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=k)
            clusters = model.fit_predict(X)

        st.success(f"✅ {algo} Done")

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap="tab10")
        st.pyplot(fig)

        st.session_state.update({
            "model": model,
            "mode": mode,
            "encoders": encoders,
            "input_cols": input_cols,
            "pca": pca
        })


# ---------------- USER INPUT ---------------- #
if "model" in st.session_state:

    st.subheader("🧠 Test Input")

    user_data = []

    for col in st.session_state["input_cols"]:
        if col in st.session_state["encoders"]:
            val = st.selectbox(col, st.session_state["encoders"][col].classes_)
        else:
            val = st.number_input(col, value=0.0)
        user_data.append(val)

    if st.button("🔮 Predict"):

        user_df = pd.DataFrame([user_data], columns=st.session_state["input_cols"])

        for col, le in st.session_state["encoders"].items():
            user_df[col] = le.transform(user_df[col])

        if st.session_state["pca"] is not None:
            user_df = st.session_state["pca"].transform(user_df)

        model = st.session_state["model"]

        if st.session_state["mode"] == "Supervised":
            pred = model.predict(user_df)

            if st.session_state["target_encoder"]:
                pred = st.session_state["target_encoder"].inverse_transform([int(pred[0])])
                st.success(f"🔮 Prediction: {pred[0]}")
            else:
                st.success(f"🔮 Prediction: {pred[0]}")
        else:
            st.warning("⚠️ Unsupervised prediction limited")