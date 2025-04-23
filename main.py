from fastapi import FastAPI, Request
import fasttext
import pandas as pd
from scipy.spatial.distance import cosine

app = FastAPI()

ft = fasttext.load_model("cc.es.300.bin")
df = pd.read_csv("IGS - Consolidado.csv")
marcas = df["NombreProducto"].astype(str).str.lower().tolist()
vectores = [ft.get_sentence_vector(m) for m in marcas]

@app.get("/")
def read_root():
    return {"msg": "API funcionando"}

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
