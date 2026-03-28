
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Western Wear Analytics", layout="wide")
st.title("👗 Western Wear E-Commerce Analytics (Final Dashboard)")

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = pd.read_csv("data.csv")

st.subheader("Raw Data")
st.dataframe(df.head())

# Cleaning
df = df.replace("Outlier", np.nan).dropna()

# Encoding
le_dict = {}
df_encoded = df.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# ---------------- EDA ----------------
st.subheader("📊 Descriptive Analytics")

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    df['Age_Group'].value_counts().plot(kind='bar')
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    df['Purchase_Intent_30d'].value_counts().plot(kind='bar')
    st.pyplot(fig2)

# ---------------- Classification ----------------
st.subheader("🤖 Classification")

X = df_encoded.drop("Purchase_Intent_30d", axis=1)
y = df_encoded["Purchase_Intent_30d"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
st.write("Accuracy:", acc)

# Feature importance
importance = pd.Series(clf.feature_importances_, index=X.columns)
st.bar_chart(importance)

# ---------------- Clustering ----------------
st.subheader("🧠 Clustering")

kmeans = KMeans(n_clusters=4, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(X)

st.bar_chart(df_encoded['Cluster'].value_counts())

# ---------------- Regression ----------------
st.subheader("📉 Regression")

target = "Monthly_Budget"
X_reg = df_encoded.drop(target, axis=1)
y_reg = df_encoded[target]

reg = LinearRegression()
reg.fit(X_reg, y_reg)

st.write("Regression model trained")

# ---------------- Association Rules ----------------
st.subheader("🛒 Association Rules")

basket = pd.get_dummies(df[['Product_Type','Combo_Purchase','Color_Preference']])
freq = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1)

st.dataframe(rules.head())

# ---------------- Prediction ----------------
st.subheader("🎯 Predict Customer")

user_input = {}
for col in df.columns:
    if col != "Purchase_Intent_30d":
        user_input[col] = st.selectbox(col, df[col].unique())

if st.button("Predict"):
    inp = pd.DataFrame([user_input])
    for col in inp.columns:
        inp[col] = le_dict[col].transform(inp[col])
    pred = clf.predict(inp)
    res = le_dict["Purchase_Intent_30d"].inverse_transform(pred)
    st.success(res[0])

# ---------------- Bulk ----------------
st.subheader("📂 Bulk Prediction")

bulk = st.file_uploader("Upload CSV", key="bulk")
if bulk:
    new_df = pd.read_csv(bulk)
    for col in new_df.columns:
        new_df[col] = le_dict[col].transform(new_df[col])
    preds = clf.predict(new_df)
    new_df['Prediction'] = le_dict["Purchase_Intent_30d"].inverse_transform(preds)
    st.dataframe(new_df.head())
