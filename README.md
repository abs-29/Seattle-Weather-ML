# 🌤️ Seattle Weather Intelligence App

An interactive Machine Learning web dashboard that predicts weather conditions based on meteorological data using an **Ensemble Model** (Gradient Boosting + XGBoost + LightGBM).

## 🚀 Key Features
- **High Accuracy (~85%+):** Utilizes advanced feature engineering including time-lagged memory (Lag Features) and 3-day moving averages.
- **Ensemble Intelligence:** Combines multiple gradient boosting algorithms for robust multi-class weather prediction.
- **Modern Interactive UI:** Built with Gradio featuring custom glassmorphism dark styling, single-row temp parameters, and real-time confidence probability breakdowns.

## 📊 Dataset & Target Classes
- **Dataset:** Seattle Weather Dataset (Kaggle)
- **Target Classes:** Sunny, Rainy, Foggy, Drizzle, Snowy
- **Predictive Features:** Max/Min Temperature, Wind Speed, Precipitation, Month, Season, Lagged Weather Features.

## 🛠️ Local Setup & Run Instructions
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/seattle-weather-ml.git](https://github.com/YOUR_USERNAME/seattle-weather-ml.git)
   cd seattle-weather-ml