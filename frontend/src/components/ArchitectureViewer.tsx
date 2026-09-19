"use client";

import React, { useState } from "react";

export default function ArchitectureViewer() {
  const [activeTab, setActiveTab] = useState<"encoder" | "decoder" | "training">("encoder");

  return (
    <section id="architecture" className="section" style={{ background: "rgba(13, 17, 26, 0.5)" }}>
      <div className="container">
        <div className="section-header">
          <div className="badge">Neural Architecture Deep Dive</div>
          <h2 className="section-title">Vision Transformer (ViT) & Decoder</h2>
          <p className="section-subtitle">
            A modular deep-learning architecture bridging computer vision and natural language processing via multi-head cross-attention.
          </p>
        </div>

        {/* Tab Selector */}
        <div style={{ display: "flex", justifyContent: "center", gap: "0.75rem", marginBottom: "2.5rem" }}>
          <button
            type="button"
            onClick={() => setActiveTab("encoder")}
            className={`btn ${activeTab === "encoder" ? "btn-primary" : "btn-secondary"}`}
          >
            ViT Vision Encoder
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("decoder")}
            className={`btn ${activeTab === "decoder" ? "btn-primary" : "btn-secondary"}`}
          >
            Autoregressive Decoder
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("training")}
            className={`btn ${activeTab === "training" ? "btn-primary" : "btn-secondary"}`}
          >
            Training & Loss Mechanics
          </button>
        </div>

        {/* Tab 1: Vision Transformer Encoder */}
        {activeTab === "encoder" && (
          <div className="arch-pipeline">
            <div className="pipeline-step">
              <div className="step-num">01</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Input Resizing & Normalization
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Incoming image is resized to <span className="mono text-cyan">224 × 224 × 3</span> and normalized using precomputed UIT-OpenVIIC channel statistics (mean: [0.5056, 0.4806, 0.4438], std: [0.2854, 0.2765, 0.2912]).
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                Tensor: (B, 3, 224, 224)
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">02</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  ViT-Tiny Patch 16×16 Backbone (timm)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Image is partitioned into <span className="mono text-cyan">14 × 14 = 196 patches</span> of size 16×16. The pretrained Vision Transformer backbone extracts 196 visual patch feature vectors of dimension <span className="mono text-indigo">192</span>, dropping the CLS token to retain spatial fidelity.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                Shape: (B, 196, 192)
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">03</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Linear Visual Feature Projection (192 ➔ 512)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  A learnable projection layer <span className="mono text-cyan">nn.Linear(192, 512)</span> projects the ViT features into the unified model embedding dimension <span className="mono text-cyan">d_model = 512</span>, matching the Transformer Decoder.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                Shape: (B, 196, 512)
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">04</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  196 Visual Patch Tokens Ready for Cross-Attention
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  The 196 projected patch tokens serve as the <span className="text-cyan">Keys and Values</span> for all cross-attention layers in the Decoder, enabling word-level visual grounding across fine image details.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                Outputs: (B, 196, 512)
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Text Decoder */}
        {activeTab === "decoder" && (
          <div className="arch-pipeline">
            <div className="pipeline-step">
              <div className="step-num">01</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Token Embedding & Positional Encoding (10,000 × 512)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Generated token IDs are mapped via an embedding matrix of size <span className="mono text-cyan">10,000 × 512</span>. Positional embeddings (up to max 500 sequence tokens) are added to inject word order.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                y: (B, seq_len, 512)
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">02</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Causal Masked Multi-Head Self-Attention (8 Heads)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  An upper triangular mask prevents tokens from attending to subsequent future positions during training and autoregressive inference, enforcing strict causal language generation.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                mask: torch.triu()
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">03</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Multi-Head Cross-Attention (Visual Grounding)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Queries come from the Vietnamese text tokens, while Keys and Values come from the <span className="text-cyan">196 visual patch embeddings (512-dim)</span> from the ViT projection. This is where text aligns with image regions!
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                Cross-Attn(Q, K, V)
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">04</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  3× Decoder Blocks & Linear Head (512 ➔ 10,000)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  3 stacked decoder blocks with GELU feed-forward networks expanding from 512 to <span className="mono text-cyan">1024</span> dimensions. The final Linear classification head projects 512 hidden states to <span className="mono text-cyan">10,000 logits</span> for next-token prediction.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                Logits: (B, seq_len, 10000)
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Training & Loss Mechanics */}
        {activeTab === "training" && (
          <div className="arch-pipeline">
            <div className="pipeline-step">
              <div className="step-num">01</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Teacher Forcing with Dynamic Padding Mask
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  During training, ground truth caption sequences are shifted right (<span className="mono text-cyan">y_pred = seq[:-1]</span>, <span className="mono text-cyan">y_true = seq[1:]</span>). Padding tokens are ignored with <span className="mono text-cyan">CrossEntropyLoss(ignore_index=pad)</span>.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                CrossEntropyLoss
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">02</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Automatic Mixed Precision (AMP GradScaler)
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  PyTorch <span className="mono text-cyan">torch.amp.GradScaler</span> scales float16 and float32 operations to prevent underflow, boosting GPU throughput while conserving VRAM.
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                torch.amp.GradScaler
              </div>
            </div>

            <div className="pipeline-step">
              <div className="step-num">03</div>
              <div>
                <h4 style={{ fontWeight: 700, fontSize: "1.1rem", marginBottom: "0.25rem" }}>
                  Differential Learning Rates & StepLR
                </h4>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Trained with Adam optimizer using tiered learning rates: <span className="mono text-cyan">lr = 1e-4</span> for pretrained ViT backbone parameters, and <span className="mono text-cyan">lr = 1e-3</span> for projection, decoder, and output head (weight decay = <span className="mono text-cyan">1e-5</span>, StepLR scheduler with <span className="mono text-cyan">γ = 0.3</span>).
                </p>
              </div>
              <div className="mono text-muted" style={{ fontSize: "0.85rem" }}>
                StepLR(gamma=0.3, step=30)
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
