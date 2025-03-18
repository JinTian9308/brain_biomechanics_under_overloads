from flask import Flask, render_template, request, jsonify, send_from_directory
import whisper
import pyttsx3
import os
import pandas as pd
import tempfile
from core_processing_module import extract_overload_values, generate_deepseek_response, predict_and_visualize
import subprocess

app = Flask(__name__)

# Directory for generated files
GENERATED_FILES_DIR = "generated_files"
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

# Initialize global variables for Whisper and TTS
whisper_model = None
tts_engine = None
INITIAL_MESSAGE = "Hello! I can help you calculate the biomechanical response of the brain under sustained overload. Please provide Gx, Gy, and Gz values."

def initialize_models():
    """Initialize Whisper model and TTS engine."""
    global whisper_model, tts_engine

    # Load Whisper model
    try:
        whisper_model = whisper.load_model("medium")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Whisper model: {e}")
        whisper_model = None

    # Initialize TTS engine
    try:
        tts_engine = pyttsx3.init()

        # Force English voice
        voices = tts_engine.getProperty('voices')
        english_voice_found = False
        for voice in voices:
            if "en" in voice.languages or "English" in voice.name:
                tts_engine.setProperty('voice', voice.id)
                print(f"Selected English voice: {voice.name}")
                english_voice_found = True
                break

        if not english_voice_found:
            print("Warning: No English voice found. Using default voice.")

        tts_engine.setProperty('rate', 230)
        print("TTS engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize TTS engine: {e}")
        tts_engine = None

# Call the initialization function
initialize_models()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/init-chat', methods=['GET'])
def init_chat():
    print("Init-chat route called")  # Add log recording
    return jsonify({'message': INITIAL_MESSAGE})


@app.route('/process-input', methods=['POST'])
def process_input():
    """Unified route to handle both audio and text input."""
    try:
        is_audio = 'audio' in request.files
        transcription = None

        # Handle audio input
        if is_audio:
            audio_file = request.files['audio']
            print(f"Received audio file: {audio_file.filename}")

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                audio_file.save(temp_audio.name)
                temp_audio_path = temp_audio.name
                print(f"Saved temporary audio file: {temp_audio_path}")

            # Convert to standard WAV format
            converted_audio_path = temp_audio_path.replace('.wav', '_converted.wav')
            convert_to_standard_wav(temp_audio_path, converted_audio_path)
            os.remove(temp_audio_path)

            # Transcribe audio
            result = whisper_model.transcribe(converted_audio_path, language='en')
            transcription = result.get('text', '').strip()
            print(f"Transcription: {transcription}")
            os.remove(converted_audio_path)

        # Handle text input or transcription result
        else:
            transcription = request.json.get('text', '').strip()
            print(f"Text input: {transcription}")

        if not transcription:
            return jsonify({'error': 'No input provided'}), 400

        # Extract overload values
        overload_values = extract_overload_values(transcription)
        if overload_values:
            Gx, Gy, Gz = overload_values.get('gx'), overload_values.get('gy'), overload_values.get('gz')

            if None in (Gx, Gy, Gz):  # Handle missing values
                return jsonify({
                    "transcription": transcription,
                    "reply": "The extracted overload values are incomplete. Please provide valid values, e.g., Gx equals 1G, Gy equals 2G, Gz equals 3G."
                })

            df_result, fig = predict_and_visualize(Gx, Gy, Gz)
            df_result.to_csv(os.path.join(GENERATED_FILES_DIR, "prediction_results.csv"), index=False)

            # Extract calculation results
            max_mises = df_result['Mises_predicted (kPa)'].max()
            max_disp = df_result['Displacement_predicted (mm)'].max()
            max_por = df_result['Pore_pressure_predicted (kPa)'].max()

            max_mises_loc = df_result.loc[df_result['Mises_predicted (kPa)'].idxmax(), ['X', 'Y', 'Z']].tolist()
            max_disp_loc = df_result.loc[df_result['Displacement_predicted (mm)'].idxmax(), ['X', 'Y', 'Z']].tolist()
            max_por_loc = df_result.loc[df_result['Pore_pressure_predicted (kPa)'].idxmax(), ['X', 'Y', 'Z']].tolist()

            response = {
                "transcription": transcription,
                "chat_message": "The results have been generated. Please check the details on the page.",
                "calculation_results": {
                    "Maximum von Mises Stress": f"{max_mises:.2f} kPa at {max_mises_loc}",
                    "Maximum Displacement": f"{max_disp:.2f} mm at {max_disp_loc}",
                    "Maximum Pore Pressure": f"{max_por:.2f} kPa at {max_por_loc}"
                },
                "chart_html": fig.to_html(full_html=False, include_plotlyjs='cdn')
            }
        else:
            # Generate chatbot response if no overload values found
            chatbot_response = generate_deepseek_response(transcription)
            response = {
                "transcription": transcription,
                "reply": chatbot_response
            }

        return jsonify(response)
    except Exception as e:
        print(f"Error in /process-input: {e}")
        return jsonify({'error': str(e)}), 500



def convert_to_standard_wav(input_path, output_path):
    """Convert audio to standard WAV format using FFmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        print(f"Audio converted to standard WAV format: {output_path}")
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        raise

@app.route('/query', methods=['POST'])
def query_point():
    """Process querying the value of a specific point"""
    data = request.json
    x = float(data.get('x'))
    y = float(data.get('y'))
    z = float(data.get('z'))

    # Load results
    csv_path = os.path.join(GENERATED_FILES_DIR, "prediction_results.csv")
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No prediction data available'}), 404

    df_result = pd.read_csv(csv_path)

    point_data = df_result[
        (df_result['X'] == x) & (df_result['Y'] == y) & (df_result['Z'] == z)
    ]

    if not point_data.empty:
        row = point_data.iloc[0]
        return jsonify({
            'Mises': row['Mises_predicted (kPa)'],
            'Displacement': row['Displacement_predicted (mm)'],
            'PorePressure': row['Pore_pressure_predicted (kPa)']
        })
    else:
        return jsonify({'error': 'Point not found'}), 404

@app.route('/generated_files/<path:filename>')
def serve_generated_files(filename):
    """Serve generated files."""
    try:
        return send_from_directory(GENERATED_FILES_DIR, filename)
    except Exception as e:
        print(f"Error in /generated_files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download-results', methods=['GET'])
def download_results():
    """Handle downloading of the prediction results file."""
    # Set file path
    file_name = "prediction_results.csv"
    file_path = os.path.join(GENERATED_FILES_DIR, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'Results file not found'}), 404

    # Use send_from_directory to return the file and force download
    try:
        return send_from_directory(GENERATED_FILES_DIR, file_name, as_attachment=True, download_name=file_name)
    except Exception as e:
        print(f"Error during file download: {e}")
        return jsonify({'error': 'Error occurred while downloading the file'}), 500




if __name__ == '__main__':
    app.run(debug=True)