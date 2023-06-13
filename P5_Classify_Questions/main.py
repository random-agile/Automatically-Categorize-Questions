from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}

@app.get("/predict/{sentence}")
async def predict_sentence(sentence):
# charger le modèle, lire une phrase et retourner le tag
    print(sentence)
    if len(sentence)%2==0:
        tag = 'python'
    else:
        tag = 'java'
    return {"tag": tag}