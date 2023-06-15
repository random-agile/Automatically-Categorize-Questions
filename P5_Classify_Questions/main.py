from fastapi import FastAPI
from joblib import load

app = FastAPI()

# Charger le modèle à partir du fichier .pickle
with open('logistic_regression_word2vec.pickle', 'rb') as f:
    model = load(f)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}


@app.get("/predict/{sentence}")
async def predict_sentence(sentence):
    # Transformer la phrase en vecteur avec Word2Vec
    vector = model.infer_vector(sentence.split())

    # Utiliser le modèle pour prédire le tag
    tag = model.predict([vector])[0]
    return {"tags": tag}
