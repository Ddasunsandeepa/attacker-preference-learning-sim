from flask import Flask, request, jsonify, render_template
import logging
import os
import pickle
import sys
import numpy as np
import shap

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve model paths relative to this file, not the process's cwd, so the
# app works no matter where it's launched from (gunicorn, systemd, etc).
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('MODEL_DIR', '.'))


def _load_model(filename: str):
    """Load a pickled model with a clear failure message instead of a bare traceback.

    NOTE: pickle.load executes arbitrary code embedded in the file. That's fine
    here because these files are produced by our own training pipeline, but if
    this ever loads a model from an untrusted source, switch to a safer format
    (e.g. skops, or re-export coefficients/trees to JSON) before deploying.
    """
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        logger.error("Model file not found: %s", path)
        sys.exit(
            f"FATAL: could not find '{filename}' at '{path}'.\n"
            f"Run the training pipeline first, or set MODEL_DIR to point at "
            f"the folder containing the .pkl files."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


# Load BOTH models once at startup
rf_model  = _load_model('rf_model.pkl')
xgb_model = _load_model('xgb_model.pkl')

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


def extract_inputs(data: dict) -> tuple:
    """
    Pull the 7 feature values out of the request body.

    Returns (inputs, errors). A field is only added to `inputs` if it was
    present and numeric — this is what lets the "Missing field" check below
    actually fire. (Previously this used `data.get(key, 0)`, so an omitted
    field silently became 0.0 and never tripped the missing-field check.)
    """
    inputs = {}
    errors = []
    for key in FEATURES:
        raw = data.get(key)
        if raw is None:
            errors.append(f"Missing field: {key}")
            continue
        try:
            inputs[key] = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Field {key} must be a number, got {raw!r}")
    return inputs, errors


def validate_inputs(inputs: dict) -> list:
    """Range check only — discrete check is handled by check_in_distribution.
    Assumes `inputs` was already produced by extract_inputs (all present, all float)."""
    errors = []
    for key, (lo, hi) in VALID_RANGES.items():
        val = inputs.get(key)
        if val is not None and not (lo <= val <= hi):
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


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": rf_model is not None and xgb_model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Extract all 7 inputs (now correctly flags missing fields)
        inputs, extract_errors = extract_inputs(data)
        if extract_errors:
            return jsonify({"error": "; ".join(extract_errors)}), 400

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
