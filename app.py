from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pickle
import os
from dotenv import load_dotenv
# Attempt to import flask_cors gracefully
try:
    from flask_cors import CORS
    has_cors = True
except ImportError:
    has_cors = False

app = Flask(__name__)

if has_cors:
    CORS(app)

# --- SECURITY ---
# Define your secret API key here. 
# In a real enterprise app, this would be hidden in a .env file!
# Load environment variables from a .env file when present
load_dotenv()

# Read the API key from env; fall back to a default for local testing
VALID_API_KEY = os.environ.get("VALID_API_KEY")
if not VALID_API_KEY:
    VALID_API_KEY = "MY_SECRET_API_KEY_123"
    print(
        "WARNING: `VALID_API_KEY` not set in environment; using insecure default."
        " Create a .env file with VALID_API_KEY to secure the app."
    )

# --- MODEL DEFINITION ---
class SalaryModel:
    def __init__(self):
        self.base = 30000
        self.exp_coeff = 5000
        self.skill_coeff = 2000

    def predict(self, years, skill):
        return self.base + (years * self.exp_coeff) + (skill * self.skill_coeff)

# --- ROBUST PICKLE LOADING FIX ---
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
            model = ModelUnpickler(file).load()
        print(" Model loaded successfully")
except Exception as e:
    print(f"Server error occurred while loading model: {e}")
    model = None

# --- API ENDPOINTS ---

@app.route('/')
@limiter.exempt 
def home():
    return jsonify({
        "message": "Salary Predictor API is online. Authentication required for /predict.",
        "status": "SECURE"
    })

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute") 
def predict():
    # --- 1. API KEY VALIDATION BLOCK ---
    # Extract the 'Authorization' header sent by the client
    auth_header = request.headers.get("Authorization")
    
    # Check if the header exists and matches our exact Bearer token
    expected_token = f"Bearer {VALID_API_KEY}"
    if not auth_header or auth_header != expected_token:
        return jsonify({
            "error": "Unauthorized", 
            "message": "Invalid or missing API Key. Access Denied."
        }), 401 # 401 is the standard HTTP code for unauthorized access

    # --- 2. MODEL INFERENCE BLOCK ---
    if model is None:
        return jsonify({"error": "Model not available on server"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        years = float(data.get('years', 0))
        skill = int(data.get('skill', 0))

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
        print(f"Unexpected error during prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please slow down.",
        "retry_after": e.description
    }), 429

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)