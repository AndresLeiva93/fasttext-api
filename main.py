from fastapi import FastAPI, Request
import pandas as pd
import fasttext
from scipy.spatial.distance import cosine
import os
import requests

app = FastAPI()

MODEL_PATH = "cc.es.300.bin"
DRIVE_ID = "166PX6_dweymTeA7n0Syd3PMpiUZmXvmF"
CSV_PATH = "IGS - Consolidado.csv"

# Descargar el modelo si no existe
def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("⏬ Descargando modelo desde Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
        r = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Modelo descargado.")

descargar_modelo()

# Cargar modelo y datos
ft = fasttext.load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
marcas = df["NombreProducto"].astype(str).str.lower().tolist()
vectores = [ft.get_sentence_vector(m) for m in marcas]

@app.get("/")
def inicio():
    return {"msg": "API de FastText funcionando 🚀"}

@app.post("/comparar")
async def comparar(request: Request):
    data = await request.json()
    entrada = data.get("marca", "").lower()
    input_vector = ft.get_sentence_vector(entrada)
    similitudes = [1 - cosine(input_vector, v) for v in vectores]
    idx_max = similitudes.index(max(similitudes))
    return {
        "marca_original": entrada,
        "marca_parecida": marcas[idx_max],
        "similitud": round(similitudes[idx_max] * 100, 2)
    }
