from fastapi import FastAPI
from joblib import load
import numpy as np
import gensim
from gensim.models import Word2Vec

app = FastAPI()


def get_word_embeddings(words, model_word2Vec):

    embeddings = []
    for word in words:
        if word in model_word2Vec.wv.key_to_index:
            embeddings.append(model_word2Vec.wv[word])
    if not embeddings:
        # Si aucun mot n'est présent dans le modèle Word2Vec, renvoyer un vecteur nul
        embeddings.append(np.zeros(100))
    return embeddings


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
    sentence_list = [sentence]
    model_word2Vec = Word2Vec(sentence_list, vector_size=100, window=5, min_count=1, workers=12)
    # Transformer la phrase en vecteur avec Word2Vec
    sentence_embedding = [np.mean(get_word_embeddings(word, model_word2Vec), axis=0) for word in sentence_list]
    print('sentence_embedding')
    print(sentence_embedding)
    # Utiliser le modèle pour prédire le tag
    tag = model.predict(sentence_embedding)[0]
    print(tag)
    print('tag')
    return {"tags": tag}
