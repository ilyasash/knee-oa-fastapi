from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from huggingface_hub import hf_hub_download # <-- Penting
import uvicorn

# Inisialisasi FastAPI
app = FastAPI(title="Knee Osteoarthritis Prediction API")

# Konfigurasi CORS
origins = ["http://localhost", "http://localhost:7860", "https://huggingface.co", "https://*.hf.space"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unduh dan muat model dari Hugging Face Hub
MODEL_REPO = "ilyasash/kneeOA-Prediction"
try:
    print("Mengunduh model dari Hugging Face Hub...")
    model_paths = {
        'resnet50': hf_hub_download(repo_id=MODEL_REPO, filename="best_model_resnet50.h5"),
        'vgg19': hf_hub_download(repo_id=MODEL_REPO, filename="best_model_vgg19.h5"),
        'densenet121': hf_hub_download(repo_id=MODEL_REPO, filename="best_model_densenet121.h5")
    }
    print("Memuat model ke memori...")
    trained_models = {
        'resnet50': tf.keras.models.load_model(model_paths['resnet50']),
        'vgg19': tf.keras.models.load_model(model_paths['vgg19']),
        'densenet121': tf.keras.models.load_model(model_paths['densenet121'])
    }
    print("Semua model dari Hugging Face berhasil dimuat.")
except Exception as e:
    print(f"Error saat mengunduh atau memuat model: {e}")
    exit()

# ... (Semua fungsi helper Anda seperti prepare_input dan to_rgb tetap di sini) ...
def prepare_input(image_array, model_name):
    img_rgb = to_rgb(image_array)
    img_resized = cv2.resize(img_rgb, (200, 200))
    img_expanded = np.expand_dims(img_resized, axis=0)
    if model_name == 'resnet50': from tensorflow.keras.applications.resnet50 import preprocess_input
    elif model_name == 'vgg19': from tensorflow.keras.applications.vgg19 import preprocess_input
    elif model_name == 'densenet121': from tensorflow.keras.applications.densenet import preprocess_input
    else: raise ValueError("Model tidak dikenal.")
    return preprocess_input(img_expanded.copy())

def to_rgb(x):
    if len(x.shape) == 2: return np.stack([x] * 3, axis=-1)
    if len(x.shape) == 3 and x.shape[-1] == 1: return np.repeat(x, 3, axis=-1)
    return x

# ... (Endpoint API Anda seperti / dan /predict/ tetap di sini) ...
@app.get("/")
def read_root(): return {"status": "Knee OA Diagnosis API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # ... Logika prediksi Anda tidak berubah ...
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')
        img_array = np.array(img)
        probs_all_models = []
        for model_name in ['resnet50', 'vgg19', 'densenet121']:
            img_input = prepare_input(img_array, model_name)
            predictions = trained_models[model_name].predict(img_input, verbose=0)
            probs_all_models.append(predictions[0])
        probs_ensemble = np.mean(probs_all_models, axis=0)
        pred_ensemble_class = np.argmax(probs_ensemble)
        confidence = float(np.max(probs_ensemble) * 100)
        classes = [f"Knee OA Level {i}" for i in range(5)]
        prediction_label = classes[pred_ensemble_class]
        result = {"prediction": prediction_label, "confidence": f"{confidence:.2f}%", "model": "Hugging Face Ensemble"}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)