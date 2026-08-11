# 🌤️ Seattle Weather Prediction — ML Project

A machine learning project that predicts next-day weather conditions (sun, rain, drizzle, fog, snow) for Seattle using historical weather data, with an interactive Gradio dashboard for real-time predictions.

## Project Overview

This project builds and compares two classification models to predict weather type from daily weather metrics (precipitation, temperature, wind), and wraps the better-performing model in an interactive web UI.

## Requirements Fulfilled

| # | Requirement | Status |
|---|-------------|--------|
| i | Raw / Adequate Dataset | ✅ 1,461 rows of daily Seattle weather (2012–2015) |
| ii | Clear Workflow | ✅ Upload → EDA → Baseline Model → Optimized Model → Comparison → UI |
| iii | Easy UI | ✅ Interactive Gradio dashboard |
| iv | Multiple Model Incorporation | ✅ Random Forest + Gradient Boosting |
| v | Regularisation Techniques | ✅ max_depth, min_samples_split/leaf, learning_rate, subsample |
| vi | Comparison of Results | ✅ Accuracy, Precision, Recall, F1-Score + Confusion Matrix |
| vii | Sound and Precise Presentation | ✅ Documented notebook with markdown sections |

## Dataset

**Source:** Seattle daily weather data (2012–2015)
**Features:** `date`, `precipitation`, `temp_max`, `temp_min`, `wind`, `weather` (target)

## Workflow

1. **Data Upload** — Load the raw CSV dataset
2. **Exploratory Data Analysis (EDA)** — Class distribution and feature correlation analysis
3. **Baseline Model** — Random Forest Classifier + SMOTE (for class imbalance)
4. **Optimized Model** — Gradient Boosting Classifier + lag features (previous day's precipitation, temperature, wind)
5. **Model Comparison** — Accuracy, Precision, Recall, F1-Score, and confusion matrix
6. **Interactive UI** — Gradio-based dashboard for live predictions

## Models

### Baseline: Random Forest + SMOTE
- Handles class imbalance via SMOTE oversampling
- Regularized with `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`

### Optimized: Gradient Boosting + Lag Features
- Adds temporal (lag) features: previous day's precipitation, max temperature, and wind
- Regularized with `learning_rate=0.1`, `max_depth=3`, `subsample=0.8`, `min_samples_leaf=5`

## Setup & Usage

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd weather-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `weather_prediction_ml.py` in Google Colab or Jupyter, or convert it to a notebook:
```bash
jupyter nbconvert --to notebook weather_prediction_ml.py
```

Run all cells in order — the script will prompt you to upload `seattle-weather.csv` (included in this repo).

### 4. Launch the Gradio app
The last section of the script launches an interactive dashboard where you can input weather parameters (temperature, wind, precipitation, and previous-day values) and get a real-time weather prediction with confidence scores.

## Project Structure

```
weather-prediction-ml/
├── weather_prediction_ml.py   # Main ML pipeline (Colab-exported script)
├── seattle-weather.csv        # Dataset
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Results

The optimized Gradient Boosting model with lag features and regularization achieved higher accuracy and F1-score compared to the baseline Random Forest model, showing that temporal (lag) features carry meaningful predictive signal for next-day weather prediction. Detailed metrics are printed and visualized when the notebook is run.

## Tech Stack

- **Data processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (RandomForestClassifier, GradientBoostingClassifier)
- **Class imbalance handling:** imbalanced-learn (SMOTE)
- **UI:** Gradio
