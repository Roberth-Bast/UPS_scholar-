import re
import time
import nltk
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# 1. INSTANCIA DE FASTAPI Y CONFIGURACIÓN DE CORS
app = FastAPI(title="UPScholar API - Sistema Unificado")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DESCARGA DE RECURSOS NLTK Y CARGA DE DATOS
try:
    nltk.data.find('corpora/stopwords')
except LookUpError:
    nltk.download('stopwords')

print("Cargando base de datos 'dataset.csv'...")
df = pd.read_csv("dataset.csv")

# Tratamiento de nulos para evitar caídas
df['Title'] = df['Title'].fillna('')
df['Keywords'] = df['Keywords'].fillna('')
df['Abstracts'] = df['Abstracts'].fillna('')
df['Authors'] = df.get('Authors', pd.Series(['N/A'] * len(df))).fillna('N/A')

def limpiar_texto(texto):
    stopwords_list = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    texto_str = str(texto).lower().replace('\\n', ' ').replace('\n', ' ')
    limpio = re.sub(r'[^a-z0-9\s]', ' ', texto_str)
    palabras = limpio.split()
    resultado = [stemmer.stem(w) for w in palabras if w not in stopwords_list]
    return " ".join(resultado)

print("Preprocesando textos...")
df['Title_Clean'] = df['Title'].apply(limpiar_texto)
df['Keywords_Clean'] = df['Keywords'].apply(limpiar_texto)
df['Abstracts_Clean'] = df['Abstracts'].apply(limpiar_texto)

# 3. PROCESAMIENTO MATEMÁTICO - MÉTODO TRADICIONAL (DEBER 5)
vectorizador_binario_title = CountVectorizer(binary=True)
vectorizador_binario_key = CountVectorizer(binary=True)
vectorizador_tf_idf_abs = TfidfVectorizer()

X_titles = vectorizador_binario_title.fit_transform(df['Title_Clean']).toarray()
X_keywords = vectorizador_binario_key.fit_transform(df['Keywords_Clean']).toarray()
X_abstracts = vectorizador_tf_idf_abs.fit_transform(df['Abstracts_Clean'])

# Matriz ítem-ítem cruzada para las RECOMENDACIONES (Top 3)
sim_titles_db = 1 - pairwise_distances(X_titles, metric="jaccard")
sim_keywords_db = 1 - pairwise_distances(X_keywords, metric="jaccard")
sim_abstracts_db = cosine_similarity(X_abstracts)
matriz_sim_combinada_db = (sim_titles_db * 0.1) + (sim_keywords_db * 0.2) + (sim_abstracts_db * 0.7)

# 4. PROCESAMIENTO MATEMÁTICO - MÉTODO EMBEDDINGS LLM (LITERAL 4)
print("Cargando modelo SentenceTransformer...")
model_llm = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_abstracts_db = model_llm.encode(df['Abstracts'].tolist(), show_progress_bar=False)

# 5. LÓGICA DE BÚSQUEDA Y RECOMENDACIÓN (TOP 10 + TOP 3 RELACIONADOS)
def obtener_3_recomendados(idx_articulo, excluir_indices):
    similitudes = matriz_sim_combinada_db[idx_articulo].copy()
    similitudes[idx_articulo] = -1
    for idx in excluir_indices:
        similitudes[idx] = -1
        
    indices_top3 = np.argsort(similitudes)[::-1][:3]
    
    recomendados = []
    for idx in indices_top3:
        recomendados.append({
            "title": str(df.iloc[idx]['Title']),
            "authors": str(df.iloc[idx]['Authors'])
        })
    return recomendados

def buscar_metodo_tradicional(query):
    query_clean = limpiar_texto(query)
    q_title = vectorizador_binario_title.transform([query_clean]).toarray()
    q_key = vectorizador_binario_key.transform([query_clean]).toarray()
    q_abs = vectorizador_tf_idf_abs.transform([query_clean])
    
    sim_t = 1 - pairwise_distances(q_title, X_titles, metric="jaccard").flatten()
    sim_k = 1 - pairwise_distances(q_key, X_keywords, metric="jaccard").flatten()
    sim_a = cosine_similarity(q_abs, X_abstracts).flatten()
    
    sim_final = (sim_t * 0.1) + (sim_k * 0.2) + (sim_a * 0.7)
    indices_top10 = np.argsort(sim_final)[::-1][:10]
    
    resultados = []
    for idx in indices_top10:
        resultados.append({
            "index": int(idx),
            "title": str(df.iloc[idx]['Title']),
            "abstract": str(df.iloc[idx]['Abstracts']),
            "score": float(sim_final[idx]),
            "recomendados_top3": obtener_3_recomendados(idx, indices_top10)
        })
    return resultados

def buscar_metodo_llm(query):
    query_embedding = model_llm.encode([query])
    sim_final = cosine_similarity(query_embedding, embeddings_abstracts_db).flatten()
    indices_top10 = np.argsort(sim_final)[::-1][:10]
    
    resultados = []
    for idx in indices_top10:
        resultados.append({
            "index": int(idx),
            "title": str(df.iloc[idx]['Title']),
            "abstract": str(df.iloc[idx]['Abstracts']),
            "score": float(sim_final[idx]),
            "recomendados_top3": obtener_3_recomendados(idx, indices_top10)
        })
    return resultados

# 6. END-POINT DE LA API
@app.get("/buscar")
def api_buscar(q: str = Query(..., min_length=1), metodo: str = "tradicional"):
    if metodo == "llm":
        return {"resultados": buscar_metodo_llm(q)}
    else:
        return {"resultados": buscar_metodo_tradicional(q)}