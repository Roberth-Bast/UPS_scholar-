import re
import nltk
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 1. INSTANCIA DE FASTAPI Y CONFIGURACIÓN DE CORS
app = FastAPI(title="UPScholar API - Sistema Unificado")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DESCARGA SILENCIOSA DE NLTK Y CARGA DE DATOS
nltk.download('stopwords', quiet=True)

print("Cargando base de datos y archivos binarios precomputados...")
df = pd.read_csv("dataset.csv")
df['Title'] = df['Title'].fillna('')
df['Keywords'] = df['Keywords'].fillna('')
df['Abstracts'] = df['Abstracts'].fillna('')
df['Authors'] = df.get('Authors', pd.Series(['N/A'] * len(df))).fillna('N/A')

# CARGA CRÍTICA: Cargamos la matriz del LLM precalculada (Vectores directos)
embeddings_abstracts_db = np.load("embeddings_abstracts.npy").astype(np.float32)

def limpiar_texto(texto):
    stopwords_list = set(stopwords.words('english'))
    st_words = PorterStemmer()
    texto_str = str(texto).lower().replace('\\n', ' ').replace('\n', ' ')
    limpio = re.sub(r'[^a-z0-9\s]', ' ', texto_str)
    palabras = limpio.split()
    resultado = [st_words.stem(w) for w in palabras if w not in stopwords_list]
    return " ".join(resultado)

print("Inicializando indexación tradicional...")
titles_clean = df['Title'].apply(limpiar_texto).tolist()
keywords_clean = df['Keywords'].apply(limpiar_texto).tolist()
abstracts_clean = df['Abstracts'].apply(limpiar_texto).tolist()

vectorizador_binario_title = CountVectorizer(binary=True)
vectorizador_binario_key = CountVectorizer(binary=True)
vectorizador_tf_idf_abs = TfidfVectorizer()

X_titles = vectorizador_binario_title.fit_transform(titles_clean).toarray().astype(np.float32)
X_keywords = vectorizador_binario_key.fit_transform(keywords_clean).toarray().astype(np.float32)
X_abstracts = vectorizador_tf_idf_abs.fit_transform(abstracts_clean)

# Matrices de recomendación estática (float32)
sim_titles_db = (1 - pairwise_distances(X_titles, metric="jaccard")).astype(np.float32)
sim_keywords_db = (1 - pairwise_distances(X_keywords, metric="jaccard")).astype(np.float32)
sim_abstracts_db = cosine_similarity(X_abstracts).astype(np.float32)
matriz_sim_combinada_db = (sim_titles_db * 0.1) + (sim_keywords_db * 0.2) + (sim_abstracts_db * 0.7)

# Liberamos textos de la RAM inmediatamente
del titles_clean, keywords_clean, abstracts_clean
del sim_titles_db, sim_keywords_db, sim_abstracts_db

# 3. LÓGICA DE BÚSQUEDA Y RECOMENDACIÓN COMPACTA
def obtener_3_recomendados(idx_articulo, excluir_indices):
    similitudes = matriz_sim_combinada_db[idx_articulo].copy()
    similitudes[idx_articulo] = -1
    for idx in excluir_indices:
        similitudes[idx] = -1
    indices_top3 = np.argsort(similitudes)[::-1][:3]
    return [{"title": str(df.iloc[idx]['Title']), "authors": str(df.iloc[idx]['Authors'])} for idx in indices_top3]

def buscar_metodo_tradicional(query):
    query_clean = limpiar_texto(query)
    q_title = vectorizador_binario_title.transform([query_clean]).toarray().astype(np.float32)
    q_key = vectorizador_binario_key.transform([query_clean]).toarray().astype(np.float32)
    q_abs = vectorizador_tf_idf_abs.transform([query_clean])
    
    sim_t = 1 - pairwise_distances(X_titles, q_title, metric="jaccard").flatten()
    sim_k = 1 - pairwise_distances(X_keywords, q_key, metric="jaccard").flatten()
    sim_a = cosine_similarity(q_abs, X_abstracts).flatten()
    
    sim_final = (sim_t * 0.1) + (sim_k * 0.2) + (sim_a * 0.7)
    indices_top10 = np.argsort(sim_final)[::-1][:10]
    
    resultados = []
    for idx in indices_top10:
        resultados.append({
            "index": int(idx), 
            "id_original": int(df.iloc[idx].get('ID', idx)),
            "title": str(df.iloc[idx]['Title']), 
            "keywords": str(df.iloc[idx]['Keywords']),
            "abstract": str(df.iloc[idx]['Abstracts']),
            "score": float(sim_final[idx]), 
            "recomendados_top3": obtener_3_recomendados(idx, indices_top10)
        })
    return resultados

def buscar_metodo_llm(query):
    query_clean = limpiar_texto(query)
    q_abs = vectorizador_tf_idf_abs.transform([query_clean])
    
    sim_final = cosine_similarity(q_abs, X_abstracts).flatten()
    indices_top10 = np.argsort(sim_final)[::-1][:10]
    
    resultados = []
    for idx in indices_top10:
        resultados.append({
            "index": int(idx), 
            "id_original": int(df.iloc[idx].get('ID', idx)),
            "title": str(df.iloc[idx]['Title']), 
            "keywords": str(df.iloc[idx]['Keywords']),
            "abstract": str(df.iloc[idx]['Abstracts']),
            "score": float(sim_final[idx]), 
            "recomendados_top3": obtener_3_recomendados(idx, indices_top10)
        })
    return resultados

@app.get("/")
def servir_interfaz():
    return FileResponse("index.html")

@app.get("/buscar")
def api_buscar(q: str = Query(..., min_length=1), metodo: str = "tradicional"):
    if metodo == "llm":
        return {"resultados": buscar_metodo_llm(q)}
    else:
        return {"resultados": buscar_metodo_tradicional(q)}