import argparse
import joblib
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

model = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='tuned_lightgbm_18.pkl', help='Name of the model file in models/ directory')
    return parser.parse_args()

def load_model(filename):
    model_path = Path(__file__).parent / 'models' / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at {model_path}')

    with open(model_path, 'rb') as f:
        loaded_model = joblib.load(f)
    
    return loaded_model

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        features = [data.get('features')]
        print(features)

        column_names = [
            'age', 'study_hours', 'class_attendance', 'sleep_hours',
            'study_method', 'course', 'gender',
            'facility_rating', 'sleep_quality', 'exam_difficulty', 'internet_access'
        ]
        df_features = pd.DataFrame(features, columns=column_names)
        print('DataFrame for inference:\n', df_features)
        
        prediction = model.predict(df_features)
        
        result = prediction[0].tolist() if hasattr(prediction[0], 'tolist') else prediction[0]
        
        return jsonify({'prediction': result, 'status': 'success'})

    except Exception as e:
        print('Error: {e}')
        return jsonify({'error': str(e), 'status': 'fail'}), 400

def start_frontend():
    frontend_path = Path(__file__).parent / 'frontend'
    print('Starting frontend on port 3000...')
    subprocess.Popen('npm start', cwd=frontend_path, shell=True)

def start_backend():
    print('Starting API on port 5000...')
    app.run(debug=True, port=5000, use_reloader=False) 

def main():
    global model
    args = get_args()
    model = load_model(args.model)

    start_frontend()
    start_backend()

if __name__ == '__main__':
    main()