from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import random
import os

app = Flask(__name__)

import urllib.parse

def youtube_link(song_name, artist):
    query = f"{song_name} {artist}"
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"


# --- Load all 5 trained models ---
traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
models = {}
for t in traits:
    fname = f"{t}_final_model.pkl"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Model file not found: {fname}")
    with open(fname, "rb") as f:
        models[t] = pickle.load(f)

# --- Load music dataset ---
music_csv = "music_data.csv"
if not os.path.exists(music_csv):
    raise FileNotFoundError(f"Music dataset not found: {music_csv}")

df = pd.read_csv(music_csv)
# Require these columns in dataset
features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'tempo']
missing_feats = [c for c in features if c not in df.columns]
if missing_feats:
    raise ValueError(f"music_data.csv must contain these columns: {missing_feats}")

df = df[['name', 'artists'] + features].dropna().reset_index(drop=True)

# --- Helper functions ---
def predict_personality(sample_input):
    """
    sample_input: pandas DataFrame with same order and columns as your model expects.
    returns: dict trait->score (rounded to 3 decimals)
    """
    results = {}
    for t, model in models.items():
        # Ensure the input matches the model’s training feature order
        model_input = sample_input[model.feature_names_in_]
        pred = model.predict(model_input)[0]
        results[t] = round(float(pred), 3)
    return results

def interpret_personality(preds):
    summary = {}
    for trait, score in preds.items():
        if score <= 2.5:
            level = "Low"
        elif score < 3.5:
            level = "Moderate"
        else:
            level = "High"

        desc = {
            "Extraversion": {
                "Low": "You are more introverted, reflective, and reserved.",
                "Moderate": "You have a balanced social energy — outgoing at times, quiet at others.",
                "High": "You are energetic, social, and enthusiastic."
            },
            "Agreeableness": {
                "Low": "You may be more assertive or skeptical.",
                "Moderate": "You are cooperative but know when to stand your ground.",
                "High": "You are kind, empathetic, and cooperative."
            },
            "Conscientiousness": {
                "Low": "You tend to be spontaneous and flexible.",
                "Moderate": "You are fairly organized but open to change.",
                "High": "You are disciplined, reliable, and goal-oriented."
            },
            "Neuroticism": {
                "Low": "You are calm and emotionally stable.",
                "Moderate": "You experience emotions moderately.",
                "High": "You may feel emotions more intensely and often."
            },
            "Openness": {
                "Low": "You prefer familiarity and routine.",
                "Moderate": "You enjoy creativity with some structure.",
                "High": "You are imaginative, curious, and open to experiences."
            }
        }
        summary[trait] = {"level": level, "desc": desc[trait][level], "score": round(score,3)}
    return summary

def personality_to_song_score(personality, song_row):
    """
    personality: dict trait->score (0-5 scale)
    song_row: pandas Series with feature columns present
    returns numeric score (higher -> better match)
    """
    return (
        personality['Extraversion'] * (song_row['energy'] + song_row['danceability']) / 2.0
        + personality['Agreeableness'] * (song_row['valence'] + song_row['acousticness']) / 2.0
        + personality['Conscientiousness'] * (1 - abs(song_row['tempo'] - 120.0) / 120.0)
        + (5.0 - personality['Neuroticism']) * song_row['valence']
        + personality['Openness'] * song_row['instrumentalness']
    )

def recommend_songs(personality, df_all, top_k=50, sample_n=5):
    df_copy = df_all.copy()
    df_copy['score'] = df_copy.apply(lambda r: personality_to_song_score(personality, r), axis=1)
    df_sorted = df_copy.sort_values(by='score', ascending=False).reset_index(drop=True)
    top_pool = df_sorted.head(top_k)
    sample_n = min(sample_n, len(top_pool))
    recommended = top_pool.sample(sample_n, random_state=random.randint(1, 10000))
    recommended = recommended.sort_values(by='score', ascending=False).reset_index(drop=True)
    return recommended

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values in the exact order your model expects
    top_features_new = [
        'Openness_group_Low',
        'Conscientiousness_group_Low',
        'Agreeableness_group_Low',
        'respectfulness',
        'Neuroticism_group_Low',
        'compassion',
        'emotional_volatility',
        'Extraversion_group_Low'
    ]
    try:
        data = []
        for feat in top_features_new:
            raw = request.form.get(feat)
            if raw is None:
                return f"Missing input for {feat}", 400
            data.append(float(raw))
        sample_input = pd.DataFrame([data], columns=top_features_new)

        preds = predict_personality(sample_input)
        summary = interpret_personality(preds)
        return render_template('result.html', preds=preds, summary=summary)
    except Exception as e:
        return f"Error during prediction: {e}", 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        personality = {t: float(request.form.get(t, 0.0)) for t in traits}
        recs = recommend_songs(personality, df, top_k=50, sample_n=5)
        records = []
        for _, row in recs.iterrows():
            records.append({
        "name": row['name'],
        "artists": row['artists'],
        "score": row['score'],
        "youtube": youtube_link(row['name'], row['artists'])
            })
        return render_template('recommend.html', songs=records)
    except Exception as e:
        return f"Error during recommendation: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
