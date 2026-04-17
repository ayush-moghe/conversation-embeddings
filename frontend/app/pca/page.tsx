import PcaSection from "../components/PcaSection";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function PcaPage() {
  return (
    <main className="page-shell">
      <section className="aurora" aria-hidden="true" />
      <section className="panel">
        <p className="eyebrow">PCA</p>
        <h1>Visualize conversation embeddings with principal component analysis.</h1>
        <PcaSection apiBaseUrl={API_BASE_URL} />
      </section>
    </main>
  );
}
