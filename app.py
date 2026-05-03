from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pickle
import os

app = Flask(__name__)

# --- MODEL DEFINITION ---
# IMPORTANT: This class MUST be defined in app.py so pickle can find it.
# It must match the structure of the class used during training.
class SalaryModel:
    def __init__(self):
        self.base = 30000
        self.exp_coeff = 5000
        self.skill_coeff = 2000

    def predict(self, years, skill):
        return self.base + (years * self.exp_coeff) + (skill * self.skill_coeff)

# Initialize Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "30 per hour"],
    storage_uri="memory://",
)

# Load the pickled model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- API ENDPOINTS ---

@app.route('/')
@limiter.exempt 
def home():
    return jsonify({
        "message": "Salary Predictor API is online",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute") 
def predict():
    if model is None:
        return jsonify({"error": "Model file not found or corrupted on server"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'years' not in data or 'skill' not in data:
            return jsonify({"error": "Missing required fields: 'years' and 'skill'"}), 400
        
        years = float(data.get('years'))
        skill = int(data.get('skill'))

        prediction = model.predict(years, skill)

        return jsonify({
            'status': 'success',
            'input': {'years': years, 'skill': skill},
            'prediction': round(prediction, 2)
        })

    except ValueError:
        return jsonify({"error": "Invalid input types. Numbers expected."}), 400
    except Exception as e:
        return jsonify({"error": "An internal error occurred"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "You are making too many requests. Please try again later.",
        "retry_after": e.description
    }), 429

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
