import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app title
st.title("Linear Regression with Feature Selection")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataframe
    st.write("Data Preview:")
    st.write(df.head())

    # Get the columns (features) for the user to select
    columns = df.columns.tolist()
    
    # Let the user select the target variable
    target_column = st.selectbox("Select the target variable (y)", columns)

    # Let the user select the features (X)
    feature_columns = st.multiselect("Select feature columns (X)", [col for col in columns if col != target_column])

    # Check if features are selected
    if len(feature_columns) > 0:
        # Prepare the data for model
        X = df[feature_columns]
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display model coefficients
        st.write("Model Coefficients:")
        st.write(model.coef_)

        # Evaluate the model
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

        # Plotting the results (if there is only one feature)
        if len(feature_columns) == 1:
            fig, ax = plt.subplots()
            ax.scatter(X_test[feature_columns[0]], y_test, color='blue', label='True values')
            ax.scatter(X_test[feature_columns[0]], y_pred, color='red', label='Predicted values')
            ax.set_xlabel(feature_columns[0])
            ax.set_ylabel(target_column)
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Please select at least one feature.")
