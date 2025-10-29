import streamlit as st
import pickle
import pandas as pd

# --- Load all 5 trained models ---
traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
models = {}

for t in traits:
    with open(f"{t}_final_model.pkl", "rb") as f:
        models[t] = pickle.load(f)

# --- Your top features order (must match training order) ---
top_features_new = [
    "respectfulness",
    "Openness_group_Low",
    "Extraversion_group_Low",
    "Agreeableness_group_Low",
    "Conscientiousness_group_Low",
    "compassion",
    "Neuroticism_group_Low",
    "emotional_volatility"
]

# --- Function to make predictions ---
def predict_personality(sample_input):
    results = {}
    for t, model in models.items():
        pred = model.predict(sample_input)[0]
        results[t] = round(pred, 3)
    return results

# --- Helper: interpret score numerically ---
def interpret_trait(trait, score):
    if score < 2:
        level = "Low"
    elif score < 3.5:
        level = "Moderate"
    else:
        level = "High"
    
    descriptions = {
        "Extraversion": {
            "Low": "You are more reserved, quiet, and introspective.",
            "Moderate": "You balance social interaction and solitude comfortably.",
            "High": "You are energetic, talkative, and outgoing."
        },
        "Agreeableness": {
            "Low": "You may be more direct and skeptical of others.",
            "Moderate": "You are cooperative yet firm in your opinions.",
            "High": "You are kind, empathetic, and cooperative."
        },
        "Conscientiousness": {
            "Low": "You tend to be spontaneous and flexible.",
            "Moderate": "You are fairly organized but adaptable.",
            "High": "You are disciplined, responsible, and goal-oriented."
        },
        "Neuroticism": {
            "Low": "You are emotionally stable and calm under pressure.",
            "Moderate": "You experience normal emotional ups and downs.",
            "High": "You tend to be more sensitive and prone to stress."
        },
        "Openness": {
            "Low": "You prefer familiarity and practicality.",
            "Moderate": "You enjoy creativity but stay grounded.",
            "High": "You are curious, imaginative, and open to new ideas."
        }
    }
    
    return f"{level} — {descriptions[trait][level]}"

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Personality Prediction System", layout="centered")
st.title("🧠 Personality Prediction System")
st.markdown("Predict your **Big Five personality traits** based on behavior and emotional inputs.")

st.header("✨ Input Your Behavioral Indicators")
st.write("Provide approximate values or select options that best describe you:")

respectfulness = st.slider("Respectfulness (0.0 - 1.0):", 0.0, 1.0, 0.66)
compassion = st.slider("Compassion (0.0 - 1.0):", 0.0, 1.0, 0.72)
emotional_volatility = st.slider("Emotional Volatility (0.0 - 1.0):", 0.0, 1.0, 0.45)

st.markdown("#### Personality Group Indicators (1 = Low, 0 = Not Low)")
Openness_group_Low = st.selectbox("Openness Low?", [0, 1])
Extraversion_group_Low = st.selectbox("Extraversion Low?", [0, 1])
Agreeableness_group_Low = st.selectbox("Agreeableness Low?", [0, 1])
Conscientiousness_group_Low = st.selectbox("Conscientiousness Low?", [0, 1])
Neuroticism_group_Low = st.selectbox("Neuroticism Low?", [0, 1])

if st.button("🔮 Predict My Personality"):
    # Create input DataFrame
    sample_input = pd.DataFrame([[
        respectfulness,
        Openness_group_Low,
        Extraversion_group_Low,
        Agreeableness_group_Low,
        Conscientiousness_group_Low,
        compassion,
        Neuroticism_group_Low,
        emotional_volatility
    ]], columns=top_features_new)

    # Predict
    final_predictions = predict_personality(sample_input)

    st.subheader("🧩 Your Personality Profile:")
    for t, val in final_predictions.items():
        st.write(f"**{t}: {val}** → {interpret_trait(t, val)}")
