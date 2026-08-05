import gradio as gr
import pandas as pd
import numpy as np
import joblib

try:
    active_model = joblib.load('weather_model.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    active_model = None
    le = None

def predict_weather_ui(date_str, temp_max, temp_min, wind, precip, precip_lag_1, precip_lag_2):
    if active_model is None or le is None:
        err_html = "<div style='color: #ef4444; padding: 15px;'>❌ <b>Error:</b> Model file ('weather_model.pkl') or LabelEncoder ('label_encoder.pkl') not found.</div>"
        return err_html, {}, pd.DataFrame({"Error": ["Model files missing"]})
        
    try:
        date_dt = pd.to_datetime(date_str)
        month = date_dt.month
        day = date_dt.day
        dayofweek = date_dt.dayofweek
        season = (month % 12 + 3) // 3

        temp_range = float(temp_max - temp_min)
        temp_avg = float((temp_max + temp_min) / 2)
        precip_roll3 = float((precip + precip_lag_1 + precip_lag_2) / 3)

        all_possible_features = {
            'precipitation': float(precip),
            'temp_max': float(temp_max),
            'temp_min': float(temp_min),
            'wind': float(wind),
            'month': int(month),
            'day': int(day),
            'dayofweek': int(dayofweek),
            'season': int(season),
            'temp_range': temp_range,
            'temp_avg': temp_avg,
            'precip_lag_1': float(precip_lag_1),
            'precip_yesterday': float(precip_lag_1),
            'temp_max_lag_1': float(temp_max),
            'temp_max_yesterday': float(temp_max),
            'wind_lag_1': float(wind),
            'wind_yesterday': float(wind),
            'precip_lag_2': float(precip_lag_2),
            'temp_max_lag_2': float(temp_max),
            'wind_lag_2': float(wind),
            'precip_roll3': precip_roll3
        }

        if hasattr(active_model, 'feature_names_in_'):
            required_cols = list(active_model.feature_names_in_)
            input_dict = {col: [all_possible_features.get(col, 0.0)] for col in required_cols}
            input_df = pd.DataFrame(input_dict)[required_cols]
        else:
            input_df = pd.DataFrame([all_possible_features])

        pred_code = active_model.predict(input_df)[0]
        weather_name = str(le.inverse_transform([pred_code])[0]).upper()

        weather_configs = {
            'SUN': {
                'title': '☀️ SUNNY & CLEAR',
                'bg': '#1e293b',
                'border': '#38bdf8',
                'desc': 'Ideal conditions for outdoor activities and travel.'
            },
            'RAIN': {
                'title': '🌧️ RAINY DAY',
                'bg': '#1e1b4b',
                'border': '#818cf8',
                'desc': 'Moderate to heavy rainfall expected. Keep an umbrella handy!'
            },
            'FOG': {
                'title': '🌫️ FOGGY WEATHER',
                'bg': '#18181b',
                'border': '#a1a1aa',
                'desc': 'Reduced visibility expected. Drive carefully!'
            },
            'DRIZZLE': {
                'title': '🌦️ LIGHT DRIZZLE',
                'bg': '#1e293b',
                'border': '#38bdf8',
                'desc': 'Light scattered showers expected throughout the day.'
            },
            'SNOW': {
                'title': '❄️ SNOWFALL',
                'bg': '#0f172a',
                'border': '#93c5fd',
                'desc': 'Freezing temperatures with potential snow accumulation.'
            }
        }

        cfg = weather_configs.get(weather_name, {
            'title': f'🌤️ {weather_name}',
            'bg': '#1e293b',
            'border': '#6366f1',
            'desc': 'Predicted weather condition based on ensemble analytics.'
        })

        card_html = f"""
        <div style="
            background-color: {cfg['bg']};
            border-left: 5px solid {cfg['border']};
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 10px;">
            <h2 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 700;">{cfg['title']}</h2>
            <p style="margin-top: 8px; color: #94a3b8; font-size: 14px;">{cfg['desc']}</p>
        </div>
        """

        prob_dict = {}
        if hasattr(active_model, 'predict_proba'):
            probs = active_model.predict_proba(input_df)[0]
            prob_dict = {str(le.classes_[i]).upper(): float(probs[i]) for i in range(len(le.classes_))}

        summary_df = pd.DataFrame({
            "Metric": ["Max Temperature", "Min Temperature", "Wind Speed", "Precipitation", "3-Day Moving Avg"],
            "Value": [f"{temp_max} °C", f"{temp_min} °C", f"{wind} m/s", f"{precip} mm", f"{precip_roll3:.2f} mm"]
        })

        return card_html, prob_dict, summary_df

    except Exception as e:
        err_html = f"<div style='color: #ef4444; padding: 15px;'>❌ <b>Error:</b> {str(e)}</div>"
        return err_html, {}, pd.DataFrame({"Error": [str(e)]})

custom_css = """
body, div, input, textarea, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}
button.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, css=custom_css, title="Seattle Weather AI") as demo:
    gr.Markdown(
        """
        # 🌤️ Seattle Weather Intelligence App
        #### Next-Gen Machine Learning Ensemble for Weather Forecasting
        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 🎛️ Forecast Parameters")

            date_input = gr.Textbox(label="Date", value="2026-06-15", placeholder="YYYY-MM-DD")

            with gr.Row(equal_height=True):
                temp_max = gr.Slider(minimum=-10, maximum=45, value=22, label="Max Temp (°C)", scale=1)
                temp_min = gr.Slider(minimum=-10, maximum=35, value=12, label="Min Temp (°C)", scale=1)

            wind = gr.Slider(minimum=0, maximum=25, value=4.5, step=0.1, label="Wind Speed (m/s)")
            precip = gr.Number(label="Precipitation Today (mm)", value=0.0)

            with gr.Accordion("🕒 Past Weather Memory (Lag Features)", open=False):
                precip_lag_1 = gr.Number(label="Precipitation 1-Day Ago (mm)", value=0.0)
                precip_lag_2 = gr.Number(label="Precipitation 2-Days Ago (mm)", value=0.0)

            submit_btn = gr.Button("🔮 Run Weather Forecast", variant="primary", elem_classes=["primary-btn"], size="lg")

        with gr.Column(scale=5):
            gr.Markdown("### 📊 Prediction Dashboard")

            output_result = gr.HTML(value="""
            <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px dashed #475569;">
                <p style="margin: 0; color: #94a3b8; text-align: center;">Click <b>Run Weather Forecast</b> to view real-time prediction.</p>
            </div>
            """)

            output_probs = gr.Label(label="Model Confidence Distribution")
            output_table = gr.DataFrame(label="Input Metrics Summary")

    submit_btn.click(
        fn=predict_weather_ui,
        inputs=[date_input, temp_max, temp_min, wind, precip, precip_lag_1, precip_lag_2],
        outputs=[output_result, output_probs, output_table]
    )

if __name__ == "__main__":
    demo.launch()