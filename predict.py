import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Decision Tree Regressor model
with open("dream_sentiment_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Load dataset to extract feature names and sentiment mapping
df = pd.read_csv("cleaned_dream_data.csv")
target_column = "Dream_Sentiment_Score"

# Extract feature names (excluding target)
if target_column in df.columns:
    feature_names = df.drop(columns=[target_column]).columns.tolist()
else:
    feature_names = df.columns.tolist()

# Find unique sentiment values from dataset
unique_sentiments = df[target_column].unique()
sentiment_mapping = {i: label for i, label in enumerate(sorted(unique_sentiments))}

# Streamlit UI
st.title("🌙 Dream Sentiment Predictor")
st.markdown("Enter dream-related details below to predict the dream sentiment.")

# Input fields for each feature
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", min_value=float(df[feature].min()), 
                            max_value=float(df[feature].max()), value=float(df[feature].mean()))
    user_input.append(value)

# Prediction button
if st.button("Predict Dream Sentiment"):
    # Convert user input to NumPy array
    new_data_point = np.array([user_input]).reshape(1, -1)

    # Predict using the trained model
    predicted_score = loaded_model.predict(new_data_point)[0]

    # Categorizing prediction into sentiment types
    if predicted_score <= 0.3:
        result = "🌑 Nightmare"
    elif predicted_score >= 0.3:
        result = "🌟 Happy Dream"
    else:
        result = "😴 Normal Sleep"

    # Display prediction result
    st.success(f"**Predicted Sentiment: {result}**")
