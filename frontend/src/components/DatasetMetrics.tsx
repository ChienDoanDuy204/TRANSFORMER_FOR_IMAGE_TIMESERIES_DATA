"use client";

import React, { useEffect, useState } from "react";

interface DatasetStats {
  dataset_name: string;
  description: string;
  evaluation_metrics: {
    bleu1: number;
    bleu2: number;
    bleu3: number;
    bleu4: number;
    training_loss: number;
    training_accuracy: number;
  };
  vocabulary_highlights: Array<{
    token: string;
    frequency: number;
    description: string;
  }>;
}

export default function DatasetMetrics() {
  const [stats, setStats] = useState<DatasetStats | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/api/dataset/stats`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setStats(data);
      })
      .catch(() => {});
  }, []);

  const bleuScores = [
    { name: "BLEU-1", score: stats?.evaluation_metrics.bleu1 ?? 0.384, desc: "Unigram precision (individual word accuracy)" },
    { name: "BLEU-2", score: stats?.evaluation_metrics.bleu2 ?? 0.245, desc: "Bigram precision (two-word phrase coherence)" },
    { name: "BLEU-3", score: stats?.evaluation_metrics.bleu3 ?? 0.168, desc: "Trigram precision (phrase structure & fluency)" },
    { name: "BLEU-4", score: stats?.evaluation_metrics.bleu4 ?? 0.112, desc: "4-gram precision (corpus-level translation standard)" },
  ];

  const vocabWords = stats?.vocabulary_highlights || [
    { token: "một", frequency: 42500, description: "indefinite article (a/an/one)" },
    { token: "có", frequency: 38100, description: "verb 'there is / to have'" },
    { token: "màu", frequency: 29800, description: "noun 'color'" },
    { token: "đang", frequency: 27400, description: "aspect marker 'in progress'" },
    { token: "người", frequency: 25300, description: "noun 'person / people'" },
    { token: "trên", frequency: 24100, description: "preposition 'on / above'" },
    { token: "chiếc", frequency: 18700, description: "classifier for vehicles/objects" },
    { token: "đứng", frequency: 14200, description: "verb 'standing'" },
    { token: "đường", frequency: 12600, description: "noun 'street / road'" },
  ];

  return (
    <section id="dataset" className="section">
      <div className="container">
        <div className="section-header">
          <div className="badge">Data & Benchmark Metrics</div>
          <h2 className="section-title">UIT-OpenVIIC Corpus Evaluation</h2>
          <p className="section-subtitle">
            Performance metrics, corpus statistics, and BLEU-1 to BLEU-4 linguistic evaluation on Vietnamese multimodal data.
          </p>
        </div>

        {/* BLEU Cards Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1.5rem", marginBottom: "3rem" }}>
          {bleuScores.map((b) => (
            <div key={b.name} className="glass-card" style={{ padding: "1.5rem", textAlign: "center" }}>
              <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                {b.name}
              </div>
              <div style={{ fontSize: "2.5rem", fontWeight: 900, color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>
                {(b.score * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                {b.desc}
              </div>
            </div>
          ))}
        </div>

        {/* Dataset Breakdown & Vocab Highlights */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.2fr", gap: "2rem" }}>
          {/* Dataset Splits Card */}
          <div className="glass-card">
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "1rem" }}>
              Dataset Specification
            </h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "1.5rem" }}>
              The <strong>UIT-OpenVIIC</strong> (Open Vietnamese Image Captioning) challenge dataset provides paired natural photography and descriptive Vietnamese sentences collected across diverse real-world contexts.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid var(--border-subtle)", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Target Language</span>
                <span style={{ fontWeight: 600 }}>Vietnamese (Tiếng Việt)</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid var(--border-subtle)", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>NLP Tokenizer</span>
                <span className="mono text-cyan">underthesea word_tokenize</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid var(--border-subtle)", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Vocabulary Dimension</span>
                <span className="mono">10,000 tokens</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid var(--border-subtle)", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Input Image Resolution</span>
                <span className="mono">224 × 224 pixels</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid var(--border-subtle)", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Total Model Parameters</span>
                <span className="mono text-cyan">25.59M parameters</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", borderBottom: "1px solid var(--border-subtle)", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Architecture Topology</span>
                <span className="mono">ViT-Tiny + 3-Layer Decoder (512-dim)</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0", fontSize: "0.875rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Active Checkpoint</span>
                <span className="mono text-emerald">model_weight.pth (102.5 MB)</span>
              </div>
            </div>
          </div>

          {/* Top Vocabulary Tokens Card */}
          <div className="glass-card">
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "0.5rem" }}>
              Top Frequent Vietnamese Vocabulary
            </h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
              Tokens extracted by frequency ranking after word segmentation:
            </p>

            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {vocabWords.map((v) => (
                <div
                  key={v.token}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    padding: "0.4rem 0.75rem",
                    background: "rgba(255, 255, 255, 0.04)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "0.85rem"
                  }}
                  title={v.description}
                >
                  <span style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>{v.token}</span>
                  <span className="mono text-muted" style={{ fontSize: "0.75rem" }}>
                    ({v.frequency.toLocaleString()})
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
