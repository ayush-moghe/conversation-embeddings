from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import httpx
import numpy as np
import os

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"

# Use gemini-embedding-001 (free) or gemini-embedding-2-preview (paid)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Only used by gemini-embedding-001 — ignored for embedding-2-preview
GEMINI_EMBEDDING_TASK_TYPE = os.getenv("GEMINI_EMBEDDING_TASK_TYPE", "CLUSTERING")

# Prefixed into the text for both models — this is how embedding-2-preview is instructed
GEMINI_EMBEDDING_TASK_INSTRUCTION = os.getenv(
    "GEMINI_EMBEDDING_TASK_INSTRUCTION",
    (
        "Focus on personality, hobbies, interests, entertainment preferences, "
        "music taste, sports, travel interests, and recurring passions. "
        "Downweight generic workflow details, scoring metrics, and boilerplate biography text."
    ),
)

IS_EMBEDDING_V2 = "embedding-2" in GEMINI_EMBEDDING_MODEL

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup log ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"[startup] model            : {GEMINI_EMBEDDING_MODEL}")
    print(f"[startup] embedding v2     : {IS_EMBEDDING_V2}")
    print(f"[startup] taskType         : {'N/A (embedding-2 uses text prefix)' if IS_EMBEDDING_V2 else GEMINI_EMBEDDING_TASK_TYPE}")
    print(f"[startup] API key set      : {bool(api_key)}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set in the backend environment",
        )
    return api_key


def _build_text(text: str) -> str:
    """Prefix the task instruction into the text for both model versions."""
    if GEMINI_EMBEDDING_TASK_INSTRUCTION:
        return (
            f"Instruction: {GEMINI_EMBEDDING_TASK_INSTRUCTION}\n\n"
            f"Profile:\n{text}"
        )
    return text


def _build_single_request(text: str) -> dict:
    """Build a single embedContent payload."""
    model_path = f"models/{GEMINI_EMBEDDING_MODEL}"
    payload = {
        "model": model_path,
        "content": {"parts": [{"text": _build_text(text)}]},
    }
    # taskType is only valid for gemini-embedding-001
    if not IS_EMBEDDING_V2:
        payload["taskType"] = GEMINI_EMBEDDING_TASK_TYPE
    return payload


def _embed_single_text(text: str) -> list[float]:
    api_key = _get_gemini_api_key()
    model_path = f"models/{GEMINI_EMBEDDING_MODEL}"
    url = f"{GEMINI_API_URL}/{model_path}:embedContent"
    payload = _build_single_request(text)

    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embedContent failed [{exc.response.status_code}]: {exc.response.text}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini embedContent network error: {str(exc)}",
        ) from exc

    values = response.json().get("embedding", {}).get("values")
    if not values:
        raise HTTPException(status_code=502, detail="Gemini embedContent response missing values")
    return [float(v) for v in values]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    api_key = _get_gemini_api_key()
    model_path = f"models/{GEMINI_EMBEDDING_MODEL}"
    url = f"{GEMINI_API_URL}/{model_path}:batchEmbedContents"
    payload = {"requests": [_build_single_request(t) for t in texts]}

    try:
        response = httpx.post(url, params={"key": api_key}, json=payload, timeout=60.0)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(texts):
            raise HTTPException(
                status_code=502,
                detail=f"Gemini batch response length mismatch: expected {len(texts)}, got {len(embeddings)}",
            )
        return [[float(v) for v in item.get("values", [])] for item in embeddings]
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        # Batch failed — surface the real error, then fall back to single calls
        print(f"[warn] batch embed failed [{exc.response.status_code}]: {exc.response.text} — falling back to single calls")
        return [_embed_single_text(t) for t in texts]
    except httpx.HTTPError as exc:
        print(f"[warn] batch embed network error: {str(exc)} — falling back to single calls")
        return [_embed_single_text(t) for t in texts]


def _cluster(points_np: np.ndarray, sample_count: int) -> list[int]:
    if sample_count < 3:
        return [0] * sample_count
    scaled = StandardScaler().fit_transform(points_np)
    labels = DBSCAN(eps=1.15, min_samples=2).fit_predict(scaled)
    return labels.tolist()

# ── Models ────────────────────────────────────────────────────────────────────

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


class SimilarityRequest(BaseModel):
    profileA: str
    profileB: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/getEmbedding")
def generate_embedding(payload: EmbeddingRequest):
    vector = _embed_single_text(payload.conversation)
    return {"embedding": vector, "dimensions": len(vector)}


@app.post("/api/getSimilarity")
def get_similarity(payload: SimilarityRequest):
    """Return cosine similarity score between two profiles (0-1)."""
    vecs = _embed_texts([payload.profileA, payload.profileB])
    a, b = np.array(vecs[0]), np.array(vecs[1])
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    if score >= 0.90:   label = "Very High"
    elif score >= 0.80: label = "High"
    elif score >= 0.70: label = "Moderate-High"
    elif score >= 0.60: label = "Moderate"
    elif score >= 0.50: label = "Low-Moderate"
    else:               label = "Low"

    return {"similarity": round(score, 4), "percent": round(score * 100, 1), "label": label}


@app.post("/api/getEmbeddingsTsne3d")
def generate_embeddings_tsne_3d(payload: BatchEmbeddingRequest):
    if payload.tsneDimensions not in (2, 3):
        raise HTTPException(status_code=422, detail="tsneDimensions must be 2 or 3")
    if payload.tsnePerplexity is not None and payload.tsnePerplexity <= 0:
        raise HTTPException(status_code=422, detail="tsnePerplexity must be greater than 0")

    if not payload.documents:
        return {"items": [], "dimensions": 0, "count": 0, "clusterCount": 0, "tsneDimensions": payload.tsneDimensions}

    texts = [doc.conversation for doc in payload.documents]
    vectors = _embed_texts(texts)
    vectors_np = np.asarray(vectors, dtype=float)
    sample_count = len(vectors)

    effective_perplexity: float | None = None
    if sample_count == 1:
        tsne_points = [[0.0] * payload.tsneDimensions]
    else:
        requested = payload.tsnePerplexity if payload.tsnePerplexity is not None else 30.0
        effective_perplexity = max(1.0, min(float(requested), float(sample_count - 1)))
        tsne = TSNE(
            n_components=payload.tsneDimensions,
            perplexity=effective_perplexity,
            random_state=42,
            init="random",
            learning_rate="auto",
            metric="cosine",
        )
        tsne_points = tsne.fit_transform(vectors_np).tolist()

    cluster_labels = _cluster(np.asarray(tsne_points, dtype=float), sample_count)

    items = [
        {
            "filename": doc.filename,
            "embedding": emb,
            "point": pt,
            "clusterLabel": cl,
        }
        for doc, emb, pt, cl in zip(payload.documents, vectors, tsne_points, cluster_labels)
    ]

    return {
        "items": items,
        "dimensions": len(vectors[0]) if vectors else 0,
        "count": len(items),
        "clusterCount": len({l for l in cluster_labels if l >= 0}),
        "tsneDimensions": payload.tsneDimensions,
        "perplexity": effective_perplexity,
        "model": GEMINI_EMBEDDING_MODEL,
    }


@app.post("/api/getEmbeddingsPca")
def generate_embeddings_pca(payload: PcaBatchEmbeddingRequest):
    if payload.pcaDimensions not in (2, 3):
        raise HTTPException(status_code=422, detail="pcaDimensions must be 2 or 3")

    if not payload.documents:
        return {"items": [], "dimensions": 0, "count": 0, "clusterCount": 0, "pcaDimensions": payload.pcaDimensions}

    texts = [doc.conversation for doc in payload.documents]
    vectors = _embed_texts(texts)
    vectors_np = np.asarray(vectors, dtype=float)
    sample_count = len(vectors)

    effective_components = min(payload.pcaDimensions, sample_count, vectors_np.shape[1])

    if sample_count == 1:
        pca_points = [[0.0] * payload.pcaDimensions]
    else:
        pca = PCA(n_components=effective_components, random_state=42)
        projected = pca.fit_transform(vectors_np).tolist()
        if effective_components < payload.pcaDimensions:
            pad = payload.pcaDimensions - effective_components
            pca_points = [pt + [0.0] * pad for pt in projected]
        else:
            pca_points = projected

    cluster_labels = _cluster(np.asarray(pca_points, dtype=float), sample_count)

    items = [
        {
            "filename": doc.filename,
            "embedding": emb,
            "point": pt,
            "clusterLabel": cl,
        }
        for doc, emb, pt, cl in zip(payload.documents, vectors, pca_points, cluster_labels)
    ]

    return {
        "items": items,
        "dimensions": len(vectors[0]) if vectors else 0,
        "count": len(items),
        "clusterCount": len({l for l in cluster_labels if l >= 0}),
        "pcaDimensions": payload.pcaDimensions,
        "model": GEMINI_EMBEDDING_MODEL,
    }

