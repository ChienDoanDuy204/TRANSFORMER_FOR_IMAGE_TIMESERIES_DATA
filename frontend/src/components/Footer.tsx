import React from "react";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-inner">
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 800, marginBottom: "0.5rem" }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: "var(--accent-cyan)" }}></span>
            <span>Vision-Language Transformer Captioning Studio</span>
          </div>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: "480px" }}>
            PyTorch implementation of a 16x16 Patch Vision Transformer (ViT) Encoder paired with an autoregressive Causal Transformer Decoder on UIT-OpenVIIC.
          </p>
        </div>

        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
          <div className="badge">PyTorch 2.14</div>
          <div className="badge">FastAPI</div>
          <div className="badge">Next.js 15</div>
          <div className="badge">Underthesea NLP</div>
        </div>
      </div>

      <div className="container" style={{ marginTop: "2rem", paddingTop: "1.5rem", borderTop: "1px solid var(--border-subtle)", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
        <div>
          © {new Date().getFullYear()} TRANSFORMER_FOR_IMAGE_TIMESERIES_DATA. All rights reserved.
        </div>
        <div style={{ display: "flex", gap: "1.5rem" }}>
          <a href="#playground" className="nav-link">Studio</a>
          <a href="#architecture" className="nav-link">Architecture</a>
          <a href="#dataset" className="nav-link">Dataset</a>
          <a href="#api" className="nav-link">API</a>
        </div>
      </div>
    </footer>
  );
}
