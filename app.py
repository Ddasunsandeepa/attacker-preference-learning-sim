from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import shap

app = Flask(__name__)

# Load BOTH models once at startup
rf_model  = pickle.load(open('rf_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))

# SHAP explainer — TreeExplainer is fast for XGBoost
explainer = shap.TreeExplainer(xgb_model)

# ── Feature definitions ──────────────────────────────────────

FEATURES = [
    'attacker_skill',
    'attacker_aggressiveness',
    'attacker_stealth',
    'defender_detection_rate',
    'defender_patch_speed',
    'alpha',
    'beta'
]

FEATURE_LABELS = [
    'Attacker Skill',
    'Attacker Aggressiveness',
    'Attacker Stealth',
    'Detection Rate',
    'Patch Speed',
    'Alpha (learning+)',
    'Beta (forgetting-)'
]

# CRITICAL: model was trained on ONLY these discrete values.
# Any input not in this list is extrapolation — report it to the user.
VALID_VALUES = {
    'attacker_skill':          [0.1, 0.3, 0.5, 0.7, 0.9],
    'attacker_aggressiveness': [0.4, 0.7, 1.0],
    'attacker_stealth':        [0.1, 0.4, 0.7],
    'defender_detection_rate': [0.2, 0.5, 0.8],
    'defender_patch_speed':    [0.2, 0.5, 0.8],
    'alpha':                   [0.0, 0.2, 0.4],
    'beta':                    [0.05, 0.1, 0.2],
}

# Continuous bounds used for slider range validation only
VALID_RANGES = {
    'attacker_skill':          (0.0, 1.0),
    'attacker_aggressiveness': (0.0, 1.0),
    'attacker_stealth':        (0.0, 1.0),
    'defender_detection_rate': (0.0, 1.0),
    'defender_patch_speed':    (0.0, 1.0),
    'alpha':                   (0.0, 1.0),
    'beta':                    (0.0, 1.0),
}

# ── Helper functions ─────────────────────────────────────────

def check_in_distribution(inputs: dict) -> dict:
    """
    Returns whether EVERY input value matches a training value exactly.
    Uses 1e-9 tolerance to avoid float precision issues.
    In Advanced Mode, most inputs will trigger EXTRAPOLATION WARNING — expected.
    """
    out_of_range = []
    for key, valid_list in VALID_VALUES.items():
        val = inputs.get(key)
        if val is not None:
            if not any(abs(val - v) < 1e-9 for v in valid_list):
                out_of_range.append(key)

    if not out_of_range:
        return {
            "in_distribution": True,
            "status": "IN-DISTRIBUTION",
            "status_color": "#10B981",
            "note": "All inputs match training data — prediction is reliable"
        }
    else:
        readable = [k.replace('_', ' ').title() for k in out_of_range]
        return {
            "in_distribution": False,
            "status": "EXTRAPOLATION WARNING",
            "status_color": "#EF4444",
            "note": f"Input(s) not in training data: {', '.join(readable)}. Prediction may be less reliable."
        }


def validate_inputs(inputs: dict) -> list:
    """Range check only — discrete check is handled by check_in_distribution."""
    errors = []
    for key, (lo, hi) in VALID_RANGES.items():
        val = inputs.get(key)
        if val is None:
            errors.append(f"Missing field: {key}")
        elif not isinstance(val, (int, float)):
            errors.append(f"Field {key} must be a number")
        elif not (lo <= val <= hi):
            errors.append(f"{key} must be between {lo} and {hi}, got {val}")
    return errors


def get_risk_level(prediction: float) -> dict:
    if prediction < 0.10:
        return {"level": "LOW",      "color": "#10B981", "label": "Low Risk"}
    elif prediction < 0.20:
        return {"level": "MODERATE", "color": "#F59E0B", "label": "Moderate Risk"}
    elif prediction < 0.30:
        return {"level": "HIGH",     "color": "#EF4444", "label": "High Risk"}
    else:
        return {"level": "CRITICAL", "color": "#7C3AED", "label": "Critical Risk"}


def get_confidence(xgb_pred: float, rf_pred: float) -> dict:
    diff = abs(xgb_pred - rf_pred)
    if diff < 0.02:
        return {"confidence": "HIGH",   "note": "Both models agree closely"}
    elif diff < 0.05:
        return {"confidence": "MEDIUM", "note": "Minor disagreement between models"}
    else:
        return {"confidence": "LOW",    "note": "Models disagree — interpret with caution"}


def get_recommendation(inputs: dict, prediction: float) -> str:
    if inputs['defender_patch_speed'] < 0.5:
        return "Increase Patch Speed — highest ROI defender upgrade (r=0.35 with resilience)"
    elif inputs['defender_detection_rate'] < 0.5:
        return "Improve Detection Rate for earlier threat visibility"
    elif inputs['attacker_skill'] >= 0.9:
        return "Attacker skill at maximum — harden all defender layers immediately"
    elif prediction < 0.10:
        return "Configuration is well-balanced — maintain current posture"
    else:
        return "Review attacker-facing controls to reduce exposure"


# ── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Extract all 7 inputs
        inputs = {
            'attacker_skill':          float(data.get('attacker_skill', 0)),
            'attacker_aggressiveness': float(data.get('attacker_aggressiveness', 0)),
            'attacker_stealth':        float(data.get('attacker_stealth', 0)),
            'defender_detection_rate': float(data.get('defender_detection_rate', 0)),
            'defender_patch_speed':    float(data.get('defender_patch_speed', 0)),
            'alpha':                   float(data.get('alpha', 0)),
            'beta':                    float(data.get('beta', 0)),
        }

        # Validate ranges
        errors = validate_inputs(inputs)
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        # Build input array in training feature order
        input_array = np.array([[
            inputs['attacker_skill'],
            inputs['attacker_aggressiveness'],
            inputs['attacker_stealth'],
            inputs['defender_detection_rate'],
            inputs['defender_patch_speed'],
            inputs['alpha'],
            inputs['beta']
        ]])

        # Predictions from both models
        xgb_pred = float(xgb_model.predict(input_array)[0])
        rf_pred  = float(rf_model.predict(input_array)[0])

        # SHAP explanation
        # CRITICAL: expected_value can be scalar OR array depending on SHAP version
        shap_vals = explainer.shap_values(input_array)[0].tolist()
        base_val  = float(np.array(explainer.expected_value).flatten()[0])

        # Derived outputs
        risk           = get_risk_level(xgb_pred)
        confidence     = get_confidence(xgb_pred, rf_pred)
        dist           = check_in_distribution(inputs)
        recommendation = get_recommendation(inputs, xgb_pred)

        # Top SHAP driver (for Executive Summary)
        abs_shap = [abs(v) for v in shap_vals]
        top_idx  = abs_shap.index(max(abs_shap))
        top_driver = FEATURE_LABELS[top_idx]

        return jsonify({
            # Core predictions
            "xgb_prediction": round(xgb_pred, 4),
            "rf_prediction":  round(rf_pred, 4),

            # Risk classification
            "risk_level": risk["level"],
            "risk_color": risk["color"],
            "risk_label": risk["label"],

            # Model agreement
            "confidence":      confidence["confidence"],
            "confidence_note": confidence["note"],

            # Training distribution check
            "in_distribution":     dist["in_distribution"],
            "distribution_status": dist["status"],
            "distribution_color":  dist["status_color"],
            "distribution_note":   dist["note"],

            # SHAP explainability
            "shap_values":    shap_vals,
            "feature_labels": FEATURE_LABELS,
            "base_value":     round(base_val, 4),

            # Executive summary fields
            "top_driver":     top_driver,
            "recommendation": recommendation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
