import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

st.title("Diabetes Prediction System")
st.markdown("### 📊 ML Pipeline using CDC Dataset")

df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# ---------------- NAV ----------------
section = st.sidebar.radio("📌 Select Step", [
    "1️⃣ Data & EDA",
    "2️⃣ Data Cleaning",
    "3️⃣ Feature Selection",
    "4️⃣ Model Training",
    "5️⃣ Model Performance",
    "6️⃣ Prediction"
])

# ---------------- 1. EDA ----------------
if section == "1️⃣ Data & EDA":
    st.subheader("Dataset Overview")

    if st.button("Show Dataset"):
        st.dataframe(df.head())

    if st.button("Show Correlation"):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ---------------- 2. DATA CLEANING ----------------
elif section == "2️⃣ Data Cleaning":
    st.subheader("Data Cleaning & Preprocessing")

    if st.button("Check Missing Values"):
        st.write(df.isnull().sum())

    method = st.selectbox("Fill Missing Values", ["None", "Mean", "Median", "Mode"])

    if st.button("Apply Cleaning"):

        df_clean = df.copy()

        # -------- Missing Handling --------
        if method == "Mean":
            df_clean.fillna(df_clean.mean(), inplace=True)
        elif method == "Median":
            df_clean.fillna(df_clean.median(), inplace=True)
        elif method == "Mode":
            df_clean.fillna(df_clean.mode().iloc[0], inplace=True)

        st.success("✅ Missing Values Handled")

        # -------- Outlier Detection --------
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1

        outliers = ((df_clean < (Q1 - 1.5 * IQR)) |
                    (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)

        outlier_count = outliers.sum()

        st.warning(f"⚠️ Outliers Detected: {outlier_count}")

        # -------- REMOVE OUTLIERS --------
        if st.checkbox("Remove Outliers"):

            before_shape = df_clean.shape

            df_clean = df_clean[~outliers]

            after_shape = df_clean.shape

            removed = before_shape[0] - after_shape[0]

            st.success("✅ Outliers Removed Successfully!")

            st.write("📉 Rows Before:", before_shape[0])
            st.write("📈 Rows After:", after_shape[0])
            st.write(f"❌ Rows Removed: {removed}")

        else:
            st.info("Outliers not removed")

        st.session_state["clean"] = df_clean

# ---------------- 3. FEATURE ----------------
elif section == "3️⃣ Feature Selection":
    st.subheader("Feature Selection")

    df_clean = st.session_state.get("clean", df.copy())

    X = df_clean.drop("Diabetes_binary", axis=1)
    y = df_clean["Diabetes_binary"]

    method = st.selectbox("Method",
                          ["All Features", "Variance", "Information Gain"])

    k = st.slider("Select Top Features (for Info Gain)", 5, 20, 10)

    if st.button("Apply Selection"):

        if method == "All Features":
            features = list(X.columns)

        elif method == "Variance":
            selector = VarianceThreshold(0.02)
            selector.fit(X)
            features = X.columns[selector.get_support()].tolist()

        else:
            scores = mutual_info_classif(X, y)
            series = pd.Series(scores, index=X.columns)
            series = series.sort_values(ascending=False)

            st.bar_chart(series)

            features = series.head(k).index.tolist()

        st.success("Features Selected")
        st.write(features)

        st.session_state["features"] = features


# ---------------- 4. TRAINING ----------------
elif section == "4️⃣ Model Training":
    st.subheader("Model Training")

    df_clean = st.session_state.get("clean", df.copy())
    features = st.session_state.get("features", list(df.columns[:-1]))

    X = df_clean[features]
    y = df_clean["Diabetes_binary"]

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "KNN", "Logistic Regression"]
    )

    test_size = st.slider("Test Size (%)", 10, 40, 20)

    if st.button("Train Model"):

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size/100, random_state=42
        )

        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.success(f"{model_choice} trained successfully!")

        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["features"] = features
        st.session_state["preds"] = preds
        st.session_state["y_test"] = y_test

        # K-Fold
        kf = KFold(n_splits=5)
        scores = []

        for train_idx, test_idx in kf.split(X_scaled):
            m = model.__class__()
            m.fit(X_scaled[train_idx], y.iloc[train_idx])
            p = m.predict(X_scaled[test_idx])
            scores.append(accuracy_score(y.iloc[test_idx], p))

        st.session_state["kf_scores"] = scores

# ---------------- 5. PERFORMANCE ----------------
elif section == "5️⃣ Model Performance":
    st.subheader("Performance")

    if "preds" not in st.session_state:
        st.warning("Train model first")
    else:
        y_test = st.session_state["y_test"]
        preds = st.session_state["preds"]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{acc*100:.2f}%")
        col2.metric("Precision", f"{prec*100:.2f}%")
        col3.metric("Recall", f"{rec*100:.2f}%")
        col4.metric("F1 Score", f"{f1*100:.2f}%")

        # st.line_chart(st.session_state["kf_scores"])
        # -------- K-FOLD (NO SESSION STATE) --------
        # -------- K-FOLD (NO SESSION STATE) --------
    from sklearn.model_selection import KFold

    df_clean = st.session_state.get("clean", df.copy())
    features = st.session_state.get("features", list(df.columns[:-1]))

    X = df_clean[features]
    y = df_clean["Diabetes_binary"]

    scaler = st.session_state["scaler"]
    model = st.session_state["model"]

    X_scaled = scaler.transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X_scaled):
        temp_model = model.__class__()
        temp_model.fit(X_scaled[train_idx], y.iloc[train_idx])
        p = temp_model.predict(X_scaled[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], p))

# ✅ SHOW AFTER LOOP
    st.subheader("📈 K-Fold Validation")
    st.line_chart(scores)

# -------- CONFUSION MATRIX --------
    st.subheader("📊 Confusion Matrix")
    st.write(confusion_matrix(y_test, preds))

# ---------------- 6. PREDICTION ----------------
elif section == "6️⃣ Prediction":
    st.subheader("Prediction")

    if "model" not in st.session_state:
        st.warning("Train model first")
    else:
        features = st.session_state["features"]

        inputs = {}
        for f in features:
            inputs[f] = st.number_input(f, value=0.0)

        if st.button("Predict"):
            arr = np.array([list(inputs.values())])

            arr = st.session_state["scaler"].transform(arr)
            pred = st.session_state["model"].predict(arr)[0]

            if pred == 1:
                st.error("Diabetic")
            else:
                st.success("Not Diabetic")