from flask import Flask, request, jsonify
from pathlib import Path
from src.inference import load_model, load_feature_names, _to_dataframe, _align_features

MODEL_DIR = Path("opt/ml/model")
MODEL_PATH = MODEL_DIR / "penguin_classifier_model.skops"
FEATURES_PATH = MODEL_DIR / "feature_names.json"

app = Flask(__name__)

model = load_model(MODEL_PATH)
feature_order = load_feature_names(FEATURES_PATH)

def predict_payload(payload):
    #load payload as a dataframe that the model expects
    records = payload.get("instances", payload)
    df = _to_dataframe(records)
    #align features
    df_aligned = _align_features(df,feature_order)
    #predict
    pred = model.predict(df_aligned)
    return {"prediction": pred.tolist()}

@app.get("/ping")
def ping():
    return jsonify({'status':'ok'}), 200

@app.post("/invocations")
def invocations():
    payload = request.get_json()
    result = predict_payload(payload)
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)