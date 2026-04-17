"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function TopNav() {
  const pathname = usePathname();

  return (
    <header className="top-nav" aria-label="Primary">
      <div className="top-nav-inner">
        <span className="nav-brand">conversation embeddings demo</span>
        <nav className="nav-links" aria-label="Main navigation">
          <Link href="/" className={pathname === "/" ? "active" : ""}>
            Embedding
          </Link>
          <Link href="/tsne" className={pathname === "/tsne" ? "active" : ""}>
            tSNE
          </Link>
          <Link href="/pca" className={pathname === "/pca" ? "active" : ""}>
            PCA
          </Link>
        </nav>
      </div>
    </header>
  );
}
