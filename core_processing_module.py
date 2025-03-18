import os
import torch
import whisper
import pyttsx3
import re
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# Load Whisper model
try:
    whisper_model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    whisper_model = None


# Text-to-Speech Engine Initialization
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 250)
tts_engine.setProperty('voice', tts_engine.getProperty('voices')[0].id)


def extract_overload_values(transcription):
    """
    Extract overload values (Gx, Gy, Gz) from text input.
    Handles variations in formatting like "Gx = 2", "Gy is 3.5", etc.
    """
    transcription = transcription.lower().replace('equal to', '=').replace('is', '=').replace('equals', '=')
    pattern = r"(gx|gy|gz)[\s]*[=]{1}[\s]*([+-]?\d*\.?\d+)[\s]*(g|g's?)?"
    matches = re.findall(pattern, transcription, re.IGNORECASE)

    overload_values = {}
    for match in matches:
        variable, value, _ = match
        overload_values[variable.lower()] = float(value)

    return overload_values if overload_values else None


def generate_deepseek_response(prompt):
    """Generate chatbot response using DeepSeek API."""
    # Keywords used to determine relevance
    relevant_keywords = ['brain biomechanics', 'Gx', 'Gy', 'Gz', 'overload', 'stress', 'displacement', 'pressure']

    # Determine if the input contains relevant keywords, and generate a response related to the task if applicable
    if any(keyword in prompt.lower() for keyword in relevant_keywords):
        prompt = f"{prompt}\n\nPlease provide a very brief and concise response related to biomechanical overload, no more than 2-3 sentences.Please answer directly and concisely without mentioning numerical values."
    else:
        # Otherwise, generate a brief guiding response
        prompt = f"{prompt}\n\nPlease provide Gx, Gy, and Gz values."

    # DeepSeek API call logic
    try:
        # DDeepSeek API URL and request headers
        api_url = "https://api.deepseek.com/v1/chat/completions"  #
        headers = {
            "Authorization": "Bearer sk-xxxx",  # Note: Add the Bearer prefix, and api key
            "Content-Type": "application/json"
        }

        # Construct request body
        data = {
            "model": "deepseek-chat",  # deepseek-chat~V3,  deepseek-reasoner~R1
            "messages": [
                {"role": "system", "content":  (
                    "You are a helpful assistant specialized in brain biomechanics. "
                    "Provide very brief and concise responses, no more than 2-3 sentences. "
                    "Do not mention numerical values such as Gx, Gy, and Gz. "
                    "Focus only on answering the user's question concisely."
                    "You should not say 'I cannot provide Gx, Gy, and Gz values' or similar disclaimers. "
                )},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,  # Control the randomness of generated text
            "max_tokens": 200,   # Control the maximum length of generated text
        }

        # Send request
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Check if the request was successful

        # Parse response
        result = response.json()
        reply = result["choices"][0]["message"]["content"].strip()

        return reply


    except requests.exceptions.RequestException as e:

        # Print detailed error information

        print(f"Request failed: {e}")

        if 'response' in locals():
            print(f"Response status code: {response.status_code}")

            print(f"Response content: {response.content}")

        return "Sorry, I am unable to generate a response at the moment. Please try again later."

    except Exception as e:

        print(f"Unexpected error: {e}")

        return "Sorry, I am unable to generate a response at the moment. Please try again later."


def predict_and_visualize(G_x, G_y, G_z):
    """
    Predict and generate a 3D Plotly chart based on the given G_x, G_y, and G_z.
    Returns:
        - df_result: Contains the prediction results DataFrame (including coordinates and predicted values).
        - fig: Plotly Figure object (can be converted to HTML for frontend display).
    """
    # Model files
    target_models = {
        'mises': 'mises_final_model.joblib',
        'disp': 'disp_final_model.joblib',
        'por': 'por_final_model.joblib'
    }

    # Load Scaler
    scaler = joblib.load("scaler.joblib")

    # Read coordinates
    coordinate_csv = "coordinate_data.csv"
    coordinate_df = pd.read_csv(coordinate_csv)
    coordinates = coordinate_df[['x-before', 'y-before', 'z-before']]

    # Merge inputs
    new_data_df = coordinates.copy()
    new_data_df['G_x'] = float(G_x)
    new_data_df['G_y'] = float(G_y)
    new_data_df['G_z'] = float(G_z)

    # Prepare feature columns
    expected_columns = ['G_x', 'G_y', 'G_z', 'x-before', 'y-before', 'z-before']
    new_data = new_data_df[expected_columns].values

    # Perform scaling
    new_data_scaled = scaler.transform(new_data)

    # Predict mises, disp, and por sequentially
    predictions = {}
    for target, model_file in target_models.items():
        model = joblib.load(model_file)
        preds = model.predict(new_data_scaled)
        # Non-negative constraint (stress, displacement)
        if target in ['mises', 'disp']:
            preds = np.maximum(preds, 0)
        # Unit conversion for por
        if target == 'por':
            preds *= 1000
        predictions[target] = preds

    # Write prediction results back to DataFrame
    new_data_df['Mises_predicted (kPa)'] = predictions['mises']
    new_data_df['Displacement_predicted (mm)'] = predictions['disp']
    new_data_df['Pore_pressure_predicted (kPa)'] = predictions['por']

    # Rename columns for better readability
    df_result = new_data_df.rename(
        columns={
            'x-before': 'X',
            'y-before': 'Y',
            'z-before': 'Z',
        }
    )

    #  ============== Generate Plotly 3D Chart ==============
    x = df_result['X']
    y = df_result['Y']
    z = df_result['Z']

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[
            "von Mises Stress (Solid)",
            "Displacement (Solid)",
            "Pore Pressure (Solid)",
            "von Mises Stress (Transparent)",
            "Displacement (Transparent)",
            "Pore Pressure (Transparent)"
        ],
        vertical_spacing=0.15
    )

    # Solid
    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=6,
                color=df_result['Mises_predicted (kPa)'],
                colorscale='Jet',
                colorbar=dict(title="Mises (kPa)", len=0.2, x=0.27, y=0.85),
                opacity=1
            ),
            hovertemplate=(
                "<b>von Mises Stress:</b> %{marker.color:.2f} kPa"
                "<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            name="Mises Stress"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=6,
                color=df_result['Displacement_predicted (mm)'],
                colorscale='Jet',
                colorbar=dict(title="Displacement (mm)", len=0.2, x=0.625, y=0.85),
                opacity=1
            ),
            hovertemplate=(
                "<b>Displacement:</b> %{marker.color:.2f} mm"
                "<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            name="Displacement"
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=6,
                color=df_result['Pore_pressure_predicted (kPa)'],
                colorscale='Jet',
                colorbar=dict(title="Pore Pressure (kPa)", len=0.2, x=0.98, y=0.85),
                opacity=1
            ),
            hovertemplate=(
                "<b>Pore Pressure:</b> %{marker.color:.2f} kPa"
                "<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            name="Pore Pressure"
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=4,
                color=df_result['Mises_predicted (kPa)'],
                colorscale='Jet',
                opacity=0.6
            ),
            hovertemplate=(
                "<b>von Mises Stress:</b> %{marker.color:.2f} kPa"
                "<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            name="Mises Stress (Transparent)"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=4,
                color=df_result['Displacement_predicted (mm)'],
                colorscale='Jet',
                opacity=0.6
            ),
            hovertemplate=(
                "<b>Displacement:</b> %{marker.color:.2f} mm"
                "<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            name="Displacement (Transparent)"
        ),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=4,
                color=df_result['Pore_pressure_predicted (kPa)'],
                colorscale='Jet',
                opacity=0.6
            ),
            hovertemplate=(
                "<b>Pore Pressure:</b> %{marker.color:.2f} kPa"
                "<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            name="Pore Pressure (Transparent)"
        ),
        row=2, col=3
    )

    fig.update_layout(
        title=dict(
            text=(
                "3D Distribution of Predicted Responses<br>"
                f"<span style='font-size:20px;'>Overloads: G<sub>x</sub>={G_x} G, G<sub>y</sub>={G_y} G, G<sub>z</sub>={G_z} G</span>"
            ),
            x=0.5,
            font=dict(size=20, family="Arial", color="black")
        ),
        font=dict(size=14, family="Arial", color="black"),
        width=1800,
        height=900,
        margin=dict(t=100, b=50),
        showlegend=False
    )

    return df_result, fig

if __name__ == "__main__":
    print("Chatbot functions ready for Flask integration.")
