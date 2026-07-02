import pickle
import numpy as np

rf = pickle.load(open("rf_model.pkl", "rb"))
xgb = pickle.load(open("xgb_model.pkl", "rb"))

tests = {
    "Low Risk":  [0.1,0.4,0.1,0.8,0.8,0.2,0.1],
    "Balanced":  [0.5,0.7,0.4,0.5,0.5,0.2,0.1],
    "High Risk": [0.9,0.8,0.3,0.2,0.2,0.2,0.1]
}

for name, vals in tests.items():
    x = np.array([vals])

    rf_pred = rf.predict(x)[0]
    xgb_pred = xgb.predict(x)[0]

    print("="*50)
    print(name)
    print("RF :", round(rf_pred,4))
    print("XGB:", round(xgb_pred,4))