"use client";

import { ChangeEvent, FormEvent, useMemo, useState } from "react";

type EmbeddingResponse = {
  embedding: number[];
  dimensions: number;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [textInput, setTextInput] = useState("");
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [embedding, setEmbedding] = useState<number[]>([]);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  const dimensions = embedding.length;

  const preview = useMemo(() => {
    if (embedding.length === 0) {
      return "";
    }

    const rounded = embedding.slice(0, 16).map((value) => value.toFixed(5));
    return `[${rounded.join(", ")}${embedding.length > 16 ? ", ..." : ""}]`;
  }, [embedding]);

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith(".txt")) {
      setError("Please upload a .txt file.");
      event.target.value = "";
      return;
    }

    try {
      const content = await file.text();
      setTextInput(content);
      setFileName(file.name);
      setError(null);
    } catch {
      setError("Could not read the selected file.");
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const conversation = textInput.trim();
    if (!conversation) {
      setError("Please add some text or upload a .txt file first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/getEmbedding`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ conversation }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data: EmbeddingResponse = await response.json();
      setEmbedding(data.embedding ?? []);
    } catch (requestError) {
      const fallback = "Could not fetch embedding from backend.";
      setError(
        requestError instanceof Error ? `${fallback} ${requestError.message}` : fallback,
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCopyEmbedding = async () => {
    if (embedding.length === 0) {
      return;
    }

    try {
      await navigator.clipboard.writeText(JSON.stringify(embedding));
      setCopyStatus("Embedding copied to clipboard.");
      setTimeout(() => setCopyStatus(null), 2200);
    } catch {
      setCopyStatus("Copy failed. Please copy from the expanded JSON view.");
      setTimeout(() => setCopyStatus(null), 2600);
    }
  };

  return (
    <main className="page-shell">
      <section className="aurora" aria-hidden="true" />
      <section className="panel">
        <p className="eyebrow">Conversation Embeddings Demo</p>
        <h1>Generate embeddings from profile snapshot.</h1>

        <form onSubmit={handleSubmit} className="form-grid">
          <label htmlFor="conversationText" className="label">
            Input Text
          </label>
          <textarea
            id="conversationText"
            value={textInput}
            onChange={(event) => setTextInput(event.target.value)}
            placeholder="Paste a conversation, note, or transcript here..."
            rows={8}
          />

          <div className="file-row">
            <label htmlFor="txtUpload" className="upload-button">
              Upload .txt
            </label>
            <input
              id="txtUpload"
              type="file"
              accept=".txt,text/plain"
              onChange={handleFileUpload}
            />
            <span className="file-label">
              {fileName ? `Loaded: ${fileName}` : "No file selected"}
            </span>
          </div>

          <button type="submit" className="submit-button" disabled={loading}>
            {loading ? "Generating..." : "Get Embedding"}
          </button>
        </form>

        {error ? <p className="status error">{error}</p> : null}

        <section className="result-card" aria-live="polite">
          <div className="result-header">
            <h2>Embedding Output</h2>
            <span>{dimensions > 0 ? `${dimensions} dimensions` : "No vector yet"}</span>
          </div>

          {dimensions > 0 ? (
            <>
              <div className="copy-row">
                <button type="button" className="copy-button" onClick={handleCopyEmbedding}>
                  Copy Full Vector
                </button>
                {copyStatus ? <span className="copy-status">{copyStatus}</span> : null}
              </div>
              <p className="preview">{preview}</p>
              <details>
                <summary>Show full embedding array</summary>
                <pre>{JSON.stringify(embedding, null, 2)}</pre>
              </details>
            </>
          ) : (
            <p className="empty-state">
              Submit text to view the generated embedding from your FastAPI backend.
            </p>
          )}
        </section>
      </section>
    </main>
  );
}
