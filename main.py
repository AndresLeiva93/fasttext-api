from fastapi import FastAPI, Request
import pandas as pd
import fasttext
from scipy.spatial.distance import cosine
import os
import requests

app = FastAPI()

MODEL_PATH = "cc.es.300.bin"
import gzip
import shutil

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("⏬ Descargando modelo desde FastText...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz"
        r = requests.get(url, stream=True)
        with open("cc.es.300.bin.gz", "wb") as f:
            shutil.copyfileobj(r.raw, f)
        print("📦 Descomprimiendo...")
        with gzip.open("cc.es.300.bin.gz", "rb") as f_in:
            with open(MODEL_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("✅ Modelo listo.")


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

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

