# 🎵 Personality-Based Music Recommendation System

A machine learning-powered web application that predicts your Big Five personality traits and recommends music tailored to your unique personality profile.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project combines personality psychology with music recommendation by:
1. **Predicting** your Big Five personality traits based on behavioral and psychological inputs
2. **Analyzing** the relationship between personality dimensions and music preferences
3. **Recommending** songs from a curated dataset that match your personality profile

The system uses machine learning models trained on psychological and music audio feature datasets to provide personalized music recommendations.

## ✨ Features

- **🧠 Personality Assessment**: Predicts all five personality dimensions (OCEAN model)
  - Openness to Experience
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism

- **🎼 Smart Music Recommendations**: Matches songs based on:
  - Audio features (danceability, energy, valence, acousticness, tempo, instrumentalness)
  - Personality-music correlations
  - Emotional characteristics

- **📊 Detailed Insights**: Provides interpretable results with:
  - Personality trait scores (0-5 scale)
  - Level classification (Low/Moderate/High)
  - Descriptive personality profiles
  - Top 5 song recommendations with match scores

- **🎨 User-Friendly Interface**: Clean, modern web interface built with Flask

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models
- **Pickle** - Model serialization

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Jinja2** - Template engine

### Data Analysis (Notebooks)
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical graphics
- **SciPy** - Statistical analysis

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/jawhra1234/Personality_Prediction_Music.git
cd Personality_Prediction_Music
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Files
Ensure these files are present in the project directory:
- `Extraversion_final_model.pkl`
- `Agreeableness_final_model.pkl`
- `Conscientiousness_final_model.pkl`
- `Neuroticism_final_model.pkl`
- `Openness_final_model.pkl`
- `music_data.csv`

## 🚀 Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Using the App**:
   - Fill out the personality assessment form with values (typically 0-5 scale)
   - Click "Predict Personality" to get your Big Five scores
   - Review your personality profile and interpretations
   - Click "Get Music Recommendations" to receive personalized song suggestions

### Input Features

The model requires these 8 input features:
- `respectfulness` - How respectful and considerate you are
- `compassion` - Your level of empathy and compassion
- `emotional_volatility` - Emotional stability/instability
- `Openness_group_Low` - Binary indicator for low openness
- `Extraversion_group_Low` - Binary indicator for low extraversion
- `Agreeableness_group_Low` - Binary indicator for low agreeableness
- `Conscientiousness_group_Low` - Binary indicator for low conscientiousness
- `Neuroticism_group_Low` - Binary indicator for low neuroticism

## 🔬 How It Works

### 1. Personality Prediction
The system uses 5 separate trained machine learning models (one per trait) to predict personality scores based on input features. Each model outputs a score on a 0-5 scale.

### 2. Personality Interpretation
Scores are categorized into levels:
- **Low**: ≤ 2.5
- **Moderate**: 2.5 - 3.5
- **High**: > 3.5

Each level includes descriptive text explaining what the score means.

### 3. Music Recommendation Algorithm
The recommendation engine calculates a match score for each song using this formula:

```
Match Score = 
  (Extraversion × (energy + danceability) / 2) +
  (Agreeableness × (valence + acousticness) / 2) +
  (Conscientiousness × tempo_stability) +
  ((5 - Neuroticism) × valence) +
  (Openness × instrumentalness)
```

The top 50 songs are selected, and 5 are randomly sampled for variety.

## 📁 Project Structure

```
Personality_Prediction_Music/
│
├── app.py                          # Flask application (main entry point)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/                         # Trained model files
│   ├── Extraversion_final_model.pkl
│   ├── Agreeableness_final_model.pkl
│   ├── Conscientiousness_final_model.pkl
│   ├── Neuroticism_final_model.pkl
│   └── Openness_final_model.pkl
│
├── templates/                      # HTML templates
│   ├── index.html                  # Home page with input form
│   ├── result.html                 # Personality results page
│   └── recommend.html              # Music recommendations page
│
├── static/                         # Static assets
│   └── style.css                   # CSS styling
│
├── data/                           # Data files
│   ├── music_data.csv              # Music dataset with audio features
│   ├── PER_dataset.csv             # Personality dataset
│   └── processed_music_personality_data.csv
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── Untitled1.ipynb             # EDA and data analysis
│   └── Untitled2.ipynb             # Model training
│
└── visualizations/                 # Generated plots and charts
    ├── personality_distributions.png
    ├── feature_importance.png
    ├── confusion_matrices.png
    └── ... (other visualization files)
```

## 🎓 Model Details

### Training Data
- **Personality Dataset**: `PER_dataset.csv` containing Big Five personality scores and behavioral features
- **Music Dataset**: `music_data.csv` with Spotify audio features (30,000+ songs)

### Features Used
The models were trained using psychological and behavioral features that correlate with personality traits.

### Audio Features in Music Dataset
- **Danceability**: How suitable for dancing (0.0 - 1.0)
- **Energy**: Intensity and activity measure (0.0 - 1.0)
- **Valence**: Musical positiveness (0.0 - 1.0)
- **Acousticness**: Acoustic vs electronic (0.0 - 1.0)
- **Instrumentalness**: Presence of vocals (0.0 - 1.0)
- **Tempo**: Beats per minute (BPM)

### Model Performance
Visualizations of model performance are available in the project directory:
- `classification_performance.png`
- `confusion_matrices.png`
- `model_performance_comprehensive.png`

## 📊 Data Analysis

The project includes comprehensive exploratory data analysis (EDA) notebooks:

### Analyses Performed
- Univariate distributions of personality traits
- Correlation analysis between personality and music features
- Statistical hypothesis testing
- Feature importance analysis
- Bivariate and multivariate relationships

### Visualizations
Multiple visualization files are included showing:
- Audio feature distributions
- Personality trait correlations
- Music-personality relationships
- Temporal trends in music preferences
- Emotional features analysis

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more personality assessment questions
- Integrate with Spotify API for real-time music streaming
- Improve recommendation algorithm
- Add user authentication and history
- Expand music database
- Create mobile-responsive design improvements

## 📝 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- **Big Five Personality Model**: Based on the OCEAN framework in personality psychology
- **Music Audio Features**: Inspired by Spotify's audio feature analysis
- **Dataset Sources**: Personality and music preference research datasets

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ and 🎵 by the Personality Music Team**

