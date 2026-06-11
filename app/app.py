import os
import sys
import json
import pickle
import threading
import time
from flask import Flask, request, jsonify, render_template, send_file

# Resolve project path and imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "src"))

from preprocess import clean_text
from pipeline import run_pipeline

# Detect Vercel environment
IS_VERCEL = 'VERCEL' in os.environ

app_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(app_dir, 'templates'),
            static_folder=os.path.join(app_dir, 'static'))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Mutex and state management
model_lock = threading.Lock()
training_in_progress = False
training_logs = []

models_dir = os.path.join(base_dir, "models")
model_path = os.path.join(models_dir, "model.pkl")
vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
metrics_path = os.path.join(models_dir, "metrics.json")

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    with model_lock:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                print("Model and Vectorizer loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("Model files not found. Model needs training.")
        return False

# Load model at startup
load_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "training_in_progress": training_in_progress,
        "timestamp": time.time()
    }), 200

@app.route('/metrics', methods=['GET'])
def get_metrics():
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics), 200
        except Exception as e:
            return jsonify({"error": f"Failed to read metrics: {str(e)}"}), 500
    return jsonify({"error": "No metrics found. Model may not be trained yet."}), 404

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 400
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Expected JSON body."}), 400
        
    title = data.get('title', '')
    text = data.get('text', '')
    
    if not title.strip() and not text.strip():
        return jsonify({"error": "Fields 'title' or 'text' must not be empty."}), 400
        
    start_time = time.time()
    combined_input = f"{title} {text}"
    cleaned_input = clean_text(combined_input)
    
    try:
        vec_input = vectorizer.transform([cleaned_input])
        prediction_class = int(model.predict(vec_input)[0])
        probabilities = model.predict_proba(vec_input)[0]
        confidence = float(probabilities[prediction_class])
        
        label = "Real" if prediction_class == 1 else "Fake"
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "prediction": label,
            "class_label": prediction_class,
            "confidence": confidence,
            "processing_time_ms": round(processing_time, 2)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def background_retrain_task():
    global training_in_progress, training_logs
    try:
        training_logs.append(f"[{time.strftime('%H:%M:%S')}] Retraining process started.")
        if run_pipeline():
            training_logs.append(f"[{time.strftime('%H:%M:%S')}] Pipeline training complete. Reloading model files.")
            load_model()
            training_logs.append(f"[{time.strftime('%H:%M:%S')}] Model reloaded successfully.")
        else:
            training_logs.append(f"[{time.strftime('%H:%M:%S')}] Pipeline failed. See server logs.")
    except Exception as e:
        training_logs.append(f"[{time.strftime('%H:%M:%S')}] Training failed: {str(e)}")
    finally:
        training_in_progress = False

@app.route('/train', methods=['POST'])
def train():
    global training_in_progress, training_logs
    if IS_VERCEL:
        return jsonify({"error": "Model training is disabled in serverless deployment environments."}), 403

    if training_in_progress:
        return jsonify({"message": "Training already in progress."}), 409
        
    training_in_progress = True
    training_logs = []
    
    thread = threading.Thread(target=background_retrain_task, daemon=True)
    thread.start()
    
    return jsonify({
        "message": "Retraining triggered in background.",
        "status_url": "/train/status"
    }), 202

@app.route('/train/status', methods=['GET'])
def train_status():
    if IS_VERCEL:
        return jsonify({
            "in_progress": False,
            "logs": ["Model training is disabled in serverless deployment environments."]
        }), 200

    return jsonify({
        "in_progress": training_in_progress,
        "logs": training_logs
    }), 200

@app.route('/swagger.json', methods=['GET'])
def swagger_spec():
    swagger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swagger.json")
    return send_file(swagger_path)

@app.route('/docs', methods=['GET'])
def swagger_docs():
    return render_template('docs.html')

@app.route('/')
def index():
    metrics_data = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error reading metrics: {e}")
            
    return render_template(
        'index.html',
        model_loaded=(model is not None),
        metrics=metrics_data,
        is_vercel=IS_VERCEL
    )

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    return jsonify({
        "error": str(e),
        "traceback": traceback.format_exc().split('\n')
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
