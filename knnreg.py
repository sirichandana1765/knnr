import streamlit as st
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("KNN Regression")

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

st.dataframe(df.head())

if st.button("Train Model"):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = KNeighborsRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.write("R2 Score:", r2_score(y_test, preds))

    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": preds}))
