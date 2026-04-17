import TsneSection from "../components/TsneSection";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function TsnePage() {
  return (
    <main className="page-shell">
      <section className="aurora" aria-hidden="true" />
      <section className="panel">
        <p className="eyebrow">tSNE</p>
        <h1>Vizualize patterns of conversation embeddings with tSNE.</h1>
        <TsneSection apiBaseUrl={API_BASE_URL} />
      </section>
    </main>
  );
}
