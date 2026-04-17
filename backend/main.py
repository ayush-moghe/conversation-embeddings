from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize model for text to vector embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingRequest(BaseModel):
    conversation: str


@app.post("/api/getEmbedding")
def generate_embedding(payload: EmbeddingRequest):
    embedding = model.encode(payload.conversation)
    vector = embedding.tolist()
    return {"embedding": vector, "dimensions": len(vector)}



