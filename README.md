# Real-Time Assessment of Brain Biomechanical Response Under Sustained High-G Overloads

This project provides a Flask-based web application to compute and visualize the biomechanical responses of the human brain under sustained overload. By inputting Gx, Gy, and Gz values (via text or speech), the system predicts von Mises stress, displacement, and pore pressure, along with a 3D visualization.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Environment and Dependencies](#environment-and-dependencies)  
5. [Quick Start](#quick-start)  
6. [Usage Instructions](#usage-instructions)  
   - [Run the Flask Web App](#run-the-flask-web-app)  
   - [Call Core Functions Directly](#call-core-functions-directly)  
7. [File Descriptions](#file-descriptions)  
8. [Reproducibility Notes](#reproducibility-notes)  
9. [FAQ](#faq)  
10. [License](#license)  
11. [References and Acknowledgments](#references-and-acknowledgments)  
12. [Contact](#contact)

---

## Introduction

In sustained high G-overload environments (e.g., high-speed flights or space missions), the human brain may experience significant mechanical stress. This project combines finite-element-based approaches and machine learning models to predict von Mises stress, displacement, and pore pressure under tri-axial overload (Gx, Gy, Gz).

Key objectives:
1. Provide a user-friendly web interface (Flask) for interactive input of Gx, Gy, and Gz.
2. Automatically run the prediction models and display 3D visualizations.
3. Generate a downloadable CSV file with detailed prediction results.
4. Answer user questions related to brain biomechanics.

---

## Features

- **Speech Recognition**: Utilizes [OpenAI Whisper](https://github.com/openai/whisper) to handle audio inputs.  
- **Text-to-Speech**: Employs [pyttsx3](https://pypi.org/project/pyttsx3/) for English TTS.  
- **Model Predictions**: Trained machine learning models (mises, disp, por) for von Mises stress, displacement, and pore pressure.  
- **Visualization**: 3D scatter plots via Plotly to illustrate predicted distributions in the brain coordinate space.  
- **Data Download**: Users can easily download the generated CSV file with results.  
- **Point Query**: Query specific 3D coordinates for  von Mises stress/displacement/pressure predictions.

---

## Project Structure
 ```
 . 
 ├── app.py # Main Flask app, provides web endpoints
 ├── core_processing_module.py # Core logic: data extraction, prediction, 3D visualization, etc.
 ├── requirements.txt # Python dependencies (optional)
 ├── coordinate_data.csv # Brain coordinates
 ├── mises_final_model.joblib # Pre-trained model for von Mises stress
 ├── disp_final_model.joblib # Pre-trained model for displacement
 ├── por_final_model.joblib # Pre-trained model for pore pressure
 ├── scaler.joblib # Data scaler (normalization/standardization)
 ├── templates/ # Flask templates (HTML files) 
 │   └── index.html
 ├── static/ # Static assets (CSS, JS)
 ├── generated_files/ # Folder for generated CSV, images, or other outputs
 ├── .gitignore # Git ignore file
 ├── LICENSE.lic # License file
 └── README.md # This README document
 ```

---

## Environment and Dependencies

- **Operating System**: Windows 10/11
- **Python Version**: 3.9.7  
- **Hardware**:
  - CPU: Intel Core i9-13900KF  
  - GPU: NVIDIA ProArt 4080  
  - RAM: 16 GB  

For GPU acceleration, ensure you have the appropriate CUDA toolkit and a compatible PyTorch build.

### Dependency Installation

If you have a `requirements.txt`, install everything with:
```bash
pip install -r requirements.txt
```

An example requirements.txt as follows :
```
Flask==3.1.0
Werkzeug==3.1.3
Jinja2==3.1.5

numpy==1.20.3
pandas==1.3.5
scipy==1.10.1
joblib==1.2.0
scikit-learn==1.3.0
xgboost==2.1.3

torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
openai-whisper==20240930
transformers==4.47.1

pyttsx3==2.98
SpeechRecognition==3.13.0
PyAudio==0.2.14

requests==2.28.2

plotly==5.24.1
matplotlib==3.4.3
seaborn==0.13.2

python-dotenv==1.0.0
aiohttp==3.11.10

```

---

### Quick Start

1. **Clone or download this repository**:
	```bash
	git clone https://github.com/JinTian9308/brain_biomechanics_under_overloads.git
	cd YourRepoName
	```

2. **Install dependencies**:
	```bash
	pip install -r requirements.txt
	```

3. **Run the Flask app**:
	```bash
	python app.py
	```
	After starting, the console will show something like:
	```
	Running on http://127.0.0.1:5000/ ...
	```
	Open your browser and go to that address to view the web interface.

---

## Usage Instructions

### Run the Flask Web App

- **Homepage**: Go to http://127.0.0.1:5000.
- **Audio Input**: If the page has a record button, click it and speak your Gx, Gy, Gz values or your question related with brain biomechanics. Whisper transcribes them.
- **Text Input**: Enter text manually, e.g., Gx=1.0, Gy=2.0, Gz=3.0.
- **Prediction Results**: After submission,  if your input is an overloaded value, the application will calculate and display a 3D plot. You can also download the CSV file with results.If your input is a question related to brain biomechanics, the application will answer it.

### Call Core Functions Directly

If you only need the prediction logic without the Flask interface, import the function from core_processing_module.py:
```python
from core_processing_module import predict_and_visualize

df_result, fig = predict_and_visualize(G_x=1.0, G_y=2.0, G_z=3.0)

# df_result is the DataFrame containing predictions
# fig is a Plotly Figure object for 3D visualization
```
---

### File Descriptions

- **`app.py`**  
  Handles Flask routes for uploading audio, processing text input, generating CSV results, etc.

- **`core_processing_module.py`**  
 The main computational logic:
  - Extract overload values (Gx, Gy, Gz) using regular expressions
  - Pre-trained models to predict stress, displacement, and pore pressure
  - Generate 3D visualizations (Plotly)
  - Call Deepseek's API to answer users' qualitative questions

- **`coordinate_data.csv`**  
 Contains coordinate data (X, Y, Z) used for batch predictions across the brain mesh.

- **`mises_final_model.joblib` / `disp_final_model.joblib` / `por_final_model.joblib`**  
 Pre-trained models for von Mises stress, displacement, and pore pressure, respectively.

- **`scaler.joblib`**  
 A data scaler for normalizing input features so that inference uses the same transformation as training.

- **`generated_files/`** 
 Output directory for storing CSV results, converted audio files, or generated plots.

- **`templates/`** 
 Contains HTML templates index.html.

---

### Reproducibility Notes
1. **Hardware and Environment**:
   - CPU: Intel Core i9-13900KF
   - GPU: NVIDIA ProArt 4080 (with appropriate CUDA drivers)
   - RAM: 16 GB
   - Python: 3.9.7

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run the app**:
   ```bash
   python app.py
   ```
   
4. **Check outputs**:
The CSV file with predictions is saved in generated_files/, or can be downloaded from the web page.

For GPU acceleration, make sure you have the correct PyTorch build (e.g., torch==2.0.1+cu118) matching your CUDA version.

---

## FAQ
1. **Why is audio transcription failing?**
   -Ensure openai-whisper is installed and configured.
   -Check microphone permissions, or test with an existing audio file.

2. **Why is pyttsx3 not speaking?**
   -System voice libraries might be missing. Install an English TTS voice package, or try a different voice ID in pyttsx3.init().


For more issues, open a GitHub Issue or see the Contact section below.

---

## License
This project is distributed under the MIT License. Please comply with its terms and retain the copyright notice.

---

## References and Acknowledgments
- [OpenAI Whisper (MIT License)](https://github.com/openai/whisper)  
- [pyttsx3 (BSD License)](https://pypi.org/project/pyttsx3/)  
- [Plotly (MIT License)](https://pypi.org/project/plotly/)  
- Additional Python libraries: pandas, numpy, requests, etc.

Grateful to all open-source contributors who make this possible.

---

## Contact
- **Email**: jintian9308@163.com
- **GitHub**: [JinTian9308](https://github.com/JinTian9308)
- **Institution**: Bioinspired Engineering and Biomechanics Center (BEBC),Xi'an Jiaotong University

> **Disclaimer**: This project is intended for research purposes only and is not a substitute for professional medical or clinical guidance. Users assume all risks associated with using the code or any derived outputs; the authors accept no liability for any outcomes.