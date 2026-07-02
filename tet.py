import pickle
import shap

xgb = pickle.load(open("xgb_model.pkl","rb"))

explainer = shap.TreeExplainer(xgb)

print("Expected value =", explainer.expected_value)