import os
import gzip
import shutil
import requests
import fasttext
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.spatial.distance import cosine

app = FastAPI()

MODEL_PATH = "cc.es.300.bin"
CSV_PATH = "IGS - Consolidado.csv"

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("⏬ Descargando modelo desde FastText...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz"
        r = requests.get(url, stream=True)
        with open("cc.es.300.bin.gz", "wb") as f:
            shutil.copyfileobj(r.raw, f)
        print("📦 Descomprimiendo modelo...")
        with gzip.open("cc.es.300.bin.gz", "rb") as f_in:
            with open(MODEL_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("✅ Modelo listo.")

descargar_modelo()
ft = fasttext.load_model(MODEL_PATH)

df = pd.read_csv(CSV_PATH)
marca_textos = df["NombreProducto"].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower().tolist()
marca_vectores = [ft.get_sentence_vector(m) for m in marca_textos]

class MarcaEntrada(BaseModel):
    marca: str

@app.post("/comparar")
def comparar_marca(data: MarcaEntrada):
    input_vector = ft.get_sentence_vector(data.marca.lower())
    similitudes = [1 - cosine(input_vector, v) for v in marca_vectores]
    mejor = max(zip(marca_textos, similitudes), key=lambda x: x[1])
    return {"entrada": data.marca, "mejor_coincidencia": mejor[0], "similitud": round(mejor[1], 4)}

@app.get("/")
def root():
    return {"msg": "API de FastText funcionando 🚀"}


