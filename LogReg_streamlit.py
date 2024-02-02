import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler


def main():
    data = st.file_uploader("Upload your dataset in CSV format", type="csv")
    if not data:
        st.write("# Logistic Regression and Scatter Plot Analysis")

        st.write(
            """
            **Instructions:**

            1. **Upload Data:**
               - Use the "Load data in csv format" button to upload a CSV file containing your data.

            2. **Feature Weights:**
               - After uploading your data, the app will perform logistic regression and display the weights associated with each feature.

            3. **Create Scatter Plots:**
               - On the sidebar, select two features from the dropdown menus to create a scatter plot. Explore the relationship between different features by visualizing the data points.

            Enjoy exploring your data!
            """
        )
    if data:
        data = load_and_preprocess_data(data)
        coef, intercept = log_reg(data)
        weights = {}
        for i, column in enumerate(data.columns[:-1]):
            weights[column] = coef[i]
        st.write("Feature Weights:")
        for feature, weight in weights.items():
            st.markdown(
                f"<p><span style='color: red'>{feature}</span>: {weight:.03f}</p>",
                unsafe_allow_html=True,
            )

        feature1 = st.sidebar.selectbox(
            "Choose feature for x-axis", options=data.columns[:-1]
        )
        feature2 = st.sidebar.selectbox(
            "Choose feature for y-axis", options=data.columns[:-1]
        )
        if feature1 and feature2:
            if feature1 == feature2:
                st.write(
                    """
                         ### Features must be different
                         """
                )
            else:
                display_graph(data, feature1, feature2)


def load_and_preprocess_data(data):
    data = pd.read_csv(data)
    s_scaler = StandardScaler()
    data.iloc[:, :-1] = s_scaler.fit_transform(data.iloc[:, :-1])
    return data


def log_reg(data, learning_rate=0.1, epsilon=1e-6):
    features = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])

    coef = np.random.normal(-1, 1, features.shape[1])
    intercept = np.random.normal(-1, 1)

    while True:
        y_pred = 1 / (1 + np.exp(-(features @ coef + intercept)))
        error = y_pred - y
        grad_0 = error.mean()
        grad_n = (features * error.reshape(-1, 1)).mean(axis=0)
        norm = np.sqrt(np.sum(grad_n**2))

        if norm < epsilon:
            break

        coef = coef - learning_rate * grad_n
        intercept = intercept - learning_rate * grad_0

    return coef, intercept


def display_graph(data, feature1, feature2):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=feature1, y=feature2, hue=data.iloc[:, -1])
    plt.title(f"{feature1} vs. {feature2}")
    plt.xlabel(f"{feature1}")
    plt.ylabel(f"{feature2}")
    plt.legend(labels=("Target", "Other"))
    st.pyplot(fig)


if __name__ == "__main__":
    main()
