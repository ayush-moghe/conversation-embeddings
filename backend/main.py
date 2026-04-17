from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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


class DocumentEmbeddingRequest(BaseModel):
    filename: str
    conversation: str


class BatchEmbeddingRequest(BaseModel):
    documents: list[DocumentEmbeddingRequest]
    tsneDimensions: int = 3
    tsnePerplexity: float | None = None


class PcaBatchEmbeddingRequest(BaseModel):
    documents: list[DocumentEmbeddingRequest]
    pcaDimensions: int = 3


@app.post("/api/getEmbedding")
def generate_embedding(payload: EmbeddingRequest):
    embedding = model.encode(payload.conversation)
    vector = embedding.tolist()
    return {"embedding": vector, "dimensions": len(vector)}


@app.post("/api/getEmbeddingsTsne3d")
def generate_embeddings_tsne_3d(payload: BatchEmbeddingRequest):
    if payload.tsneDimensions not in (2, 3):
        raise HTTPException(status_code=422, detail="tsneDimensions must be 2 or 3")
    if payload.tsnePerplexity is not None and payload.tsnePerplexity <= 0:
        raise HTTPException(status_code=422, detail="tsnePerplexity must be greater than 0")

    if not payload.documents:
        return {
            "items": [],
            "dimensions": 0,
            "count": 0,
            "clusterCount": 0,
            "tsneDimensions": payload.tsneDimensions,
        }

    texts = [doc.conversation for doc in payload.documents]
    vectors = model.encode(texts)
    vectors_np = np.asarray(vectors, dtype=float)
    embeddings = vectors_np.tolist()

    sample_count = len(embeddings)
    effective_perplexity: float | None = None
    if sample_count == 1:
        tsne_points = [[0.0 for _ in range(payload.tsneDimensions)]]
    else:
        max_perplexity = sample_count - 1
        requested_perplexity = payload.tsnePerplexity if payload.tsnePerplexity is not None else 30.0
        effective_perplexity = max(1.0, min(float(requested_perplexity), float(max_perplexity)))
        tsne = TSNE(
            n_components=payload.tsneDimensions,
            perplexity=effective_perplexity,
            random_state=42,
            init="random",
            learning_rate="auto",
            metric="cosine",
        )
        tsne_points = tsne.fit_transform(vectors_np).tolist()

    tsne_points_np = np.asarray(tsne_points, dtype=float)
    if sample_count < 3:
        cluster_labels = [0 for _ in range(sample_count)]
    else:
        scaled_points = StandardScaler().fit_transform(tsne_points_np)
        cluster_model = DBSCAN(eps=1.15, min_samples=2)
        cluster_labels = cluster_model.fit_predict(scaled_points).tolist()

    items = []
    for doc, embedding, point, cluster_label in zip(payload.documents, embeddings, tsne_points, cluster_labels):
        items.append(
            {
                "filename": doc.filename,
                "embedding": embedding,
                "point": point,
                "clusterLabel": cluster_label,
            }
        )

    unique_clusters = sorted({label for label in cluster_labels if label >= 0})

    return {
        "items": items,
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "count": len(items),
        "clusterCount": len(unique_clusters),
        "tsneDimensions": payload.tsneDimensions,
        "perplexity": effective_perplexity,
    }


@app.post("/api/getEmbeddingsPca")
def generate_embeddings_pca(payload: PcaBatchEmbeddingRequest):
    if payload.pcaDimensions not in (2, 3):
        raise HTTPException(status_code=422, detail="pcaDimensions must be 2 or 3")

    if not payload.documents:
        return {
            "items": [],
            "dimensions": 0,
            "count": 0,
            "clusterCount": 0,
            "pcaDimensions": payload.pcaDimensions,
        }

    texts = [doc.conversation for doc in payload.documents]
    vectors = model.encode(texts)
    vectors_np = np.asarray(vectors, dtype=float)
    embeddings = vectors_np.tolist()

    sample_count = len(embeddings)
    effective_components = min(payload.pcaDimensions, sample_count, vectors_np.shape[1])

    if sample_count == 1:
        pca_points = [[0.0 for _ in range(payload.pcaDimensions)]]
    else:
        pca = PCA(n_components=effective_components, random_state=42)
        projected = pca.fit_transform(vectors_np).tolist()
        # Zero-pad when data has fewer available components than requested (e.g. very small sample count).
        if effective_components < payload.pcaDimensions:
            pca_points = [point + [0.0] * (payload.pcaDimensions - effective_components) for point in projected]
        else:
            pca_points = projected

    pca_points_np = np.asarray(pca_points, dtype=float)
    if sample_count < 3:
        cluster_labels = [0 for _ in range(sample_count)]
    else:
        scaled_points = StandardScaler().fit_transform(pca_points_np)
        cluster_model = DBSCAN(eps=1.15, min_samples=2)
        cluster_labels = cluster_model.fit_predict(scaled_points).tolist()

    items = []
    for doc, embedding, point, cluster_label in zip(payload.documents, embeddings, pca_points, cluster_labels):
        items.append(
            {
                "filename": doc.filename,
                "embedding": embedding,
                "point": point,
                "clusterLabel": cluster_label,
            }
        )

    unique_clusters = sorted({label for label in cluster_labels if label >= 0})

    return {
        "items": items,
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "count": len(items),
        "clusterCount": len(unique_clusters),
        "pcaDimensions": payload.pcaDimensions,
    }



