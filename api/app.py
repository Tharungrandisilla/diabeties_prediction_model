from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load model, scaler, feature names once at startup ──
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE, '..', 'models', 'best_model.pkl')
SCALER_PATH       = os.path.join(BASE, '..', 'models', 'scaler.pkl')
FEAT_PATH         = os.path.join(BASE, '..', 'models', 'feature_names.pkl')

model         = joblib.load(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEAT_PATH)

print("Model loaded successfully!")
print(f"Expected features: {feature_names}")


# ── Home route — just to confirm API is running ──
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message'  : 'HealthPredict AI is running!',
        'model'    : 'XGBoost',
        'version'  : '1.0',
        'endpoint' : 'POST /predict'
    })


# ── Prediction route — the main endpoint ──
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Step 2: Build feature array in correct order
        features = []
        missing  = []

        for feat in feature_names:
            if feat not in data:
                missing.append(feat)
            else:
                features.append(float(data[feat]))

        if missing:
            return jsonify({
                'error'           : 'Missing features',
                'missing_features': missing
            }), 400

        # Step 3: Scale the input exactly like training data
        features_array  = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Step 4: Get prediction and probability
        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        prob_no_diabetes = round(float(probability[0]) * 100, 1)
        prob_diabetes    = round(float(probability[1]) * 100, 1)

        # Step 5: Build human-readable result
        if prediction == 1:
            risk_level = 'HIGH RISK'
            advice     = 'This patient shows high risk of diabetes. Recommend immediate medical consultation.'
        else:
            if prob_diabetes > 30:
                risk_level = 'MODERATE RISK'
                advice     = 'Low prediction but moderate probability. Recommend lifestyle monitoring.'
            else:
                risk_level = 'LOW RISK'
                advice     = 'Patient shows low risk of diabetes. Recommend regular health checkups.'

        # Step 6: Return result
        return jsonify({
            'prediction'        : int(prediction),
            'result'            : 'Diabetes Detected' if prediction == 1 else 'No Diabetes',
            'risk_level'        : risk_level,
            'probability'       : {
                'no_diabetes'   : prob_no_diabetes,
                'diabetes'      : prob_diabetes
            },
            'advice'            : advice,
            'model_used'        : 'XGBoost',
            'features_received' : len(features)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Health check route ──
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'healthy',
        'model'   : 'loaded',
        'features': len(feature_names)
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  HealthPredict AI — Flask API Starting...")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)