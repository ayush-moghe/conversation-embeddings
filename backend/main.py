from fastapi import FastAPI, UploadFile
from scikitlearn.manifold import TSNE
app = FastAPI()

@app.post("api/getEmbedding")
def generate_embedding(file: UploadFile):
    return NotImplementedError("This endpoint is not implemented yet")


@app.get("api/tSNE")
def get_tSNE():
    return NotImplementedError("This endpoint is not implemented yet")

