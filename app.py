from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load all 5 models
traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
models = {}
for t in traits:
    with open(f"{t}_final_model.pkl", "rb") as f:
        models[t] = pickle.load(f)

# Top features (order must match model training)
top_features_new = [
    "respectfulness", "Openness_group_Low", "Extraversion_group_Low",
    "Agreeableness_group_Low", "Conscientiousness_group_Low",
    "compassion", "Neuroticism_group_Low", "emotional_volatility"
]

# Personality interpretation
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
                "Low": "You tend to be straightforward and less concerned with pleasing others.",
                "Moderate": "You are kind but also assertive when needed.",
                "High": "You are empathetic, kind, and cooperative."
            },
            "Conscientiousness": {
                "Low": "You prefer flexibility over structure and may act spontaneously.",
                "Moderate": "You are fairly organized and reliable.",
                "High": "You are disciplined, organized, and goal-focused."
            },
            "Neuroticism": {
                "Low": "You are calm, emotionally stable, and resilient.",
                "Moderate": "You experience occasional emotional ups and downs.",
                "High": "You are sensitive and may experience emotions more intensely."
            },
            "Openness": {
                "Low": "You are practical, realistic, and prefer familiarity.",
                "Moderate": "You are open to some new experiences but value routine too.",
                "High": "You are creative, curious, and open-minded."
            }
        }
        summary[trait] = {"level": level, "desc": desc[trait][level]}
    return summary

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs from form
    data = [float(request.form.get(feat)) for feat in top_features_new]
    sample_input = pd.DataFrame([data], columns=top_features_new)
    
    # Predict each trait
    preds = {}
    for t, model in models.items():
        preds[t] = round(model.predict(sample_input)[0], 3)
    
    # Interpret results
    summary = interpret_personality(preds)
    
    return render_template("result.html", predictions=preds, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
