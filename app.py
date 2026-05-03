from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import pickle
import os
import sys

app = Flask(__name__)
CORS(app) # Enable CORS for frontend integration

# --- MODEL DEFINITION ---
# This class must match the structure of the one used during training.
class SalaryModel:
    def __init__(self):
        self.base = 30000
        self.exp_coeff = 5000
        self.skill_coeff = 2000

    def predict(self, years, skill):
        return self.base + (years * self.exp_coeff) + (skill * self.skill_coeff)

# --- ROBUST PICKLE LOADING FIX ---
# This CustomUnpickler solves the 'AttributeError: module __main__ has no attribute' 
# error that happens specifically on Render/Gunicorn.
class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SalaryModel':
            return SalaryModel
        return super().find_class(module, name)

# Initialize Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "30 per hour"],
    storage_uri="memory://",
)

# Load the pickled model
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            # Use our custom unpickler instead of standard pickle.load
            model = ModelUnpickler(file).load()
        print("✅ Model loaded successfully using ModelUnpickler")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- API ENDPOINTS ---

@app.route('/')
@limiter.exempt 
def home():
    return jsonify({
        "message": "Salary Predictor API is online",
        "status": "online",
        "deployment_url": "https://project-w1nx.onrender.com/",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "python_version": sys.version
    }), 200

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute") 
def predict():
    if model is None:
        return jsonify({"error": "Model not available on server"}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        years = float(data.get('years', 0))
        skill = int(data.get('skill', 0))

        # Input validation
        if years < 0 or skill < 0:
            return jsonify({"error": "Values cannot be negative"}), 400

        prediction = model.predict(years, skill)

        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 2),
            'currency': 'USD'
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid data format. Numbers expected."}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please slow down.",
        "retry_after": e.description
    }), 429

if __name__ == '__main__':
    # Local development settings
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
