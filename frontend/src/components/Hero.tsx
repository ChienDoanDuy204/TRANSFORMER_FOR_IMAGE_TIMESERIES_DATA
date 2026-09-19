"use client";

import React from "react";

export default function Hero() {
  return (
    <section className="hero">
      <div className="container">
        <div className="badge" style={{ marginBottom: "1.25rem" }}>
          <span>✨</span>
          <span>End-to-End PyTorch Vision-Language Transformer</span>
        </div>

        <h1 className="hero-title">
          Deep Vision Transformer for{" "}
          <span className="text-gradient">Vietnamese Image Captioning</span>
        </h1>

        <p className="hero-description">
          An end-to-end multimodal deep neural network combining a 16×16 Patch <strong>Vision Transformer (ViT-Tiny) Backbone</strong> with a 3-Layer <strong>Causal Transformer Decoder</strong> (512-dim embedding, 1024-dim FFN) featuring Multi-Head Cross-Attention and Vietnamese word segmentation.
        </p>

        <div className="hero-actions">
          <a href="#playground" className="btn btn-primary" style={{ padding: "0.85rem 1.8rem", fontSize: "1rem" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            <span>Launch AI Playground</span>
          </a>

          <a href="#architecture" className="btn btn-secondary" style={{ padding: "0.85rem 1.8rem", fontSize: "1rem" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
              <line x1="8" y1="21" x2="16" y2="21" />
              <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
            <span>Explore Architecture</span>
          </a>

          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="btn btn-ghost"
            style={{ padding: "0.85rem 1.4rem" }}
          >
            <span>Documentation</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </a>
        </div>

        <div className="hero-stats">
          <div className="stat-box">
            <div className="stat-value">196</div>
            <div className="stat-label">ViT 16×16 Visual Patches</div>
          </div>

          <div className="stat-box">
            <div className="stat-value">25.59M</div>
            <div className="stat-label">Model Parameters</div>
          </div>

          <div className="stat-box">
            <div className="stat-value">10,000</div>
            <div className="stat-label">Vietnamese Vocabulary Tokens</div>
          </div>

          <div className="stat-box">
            <div className="stat-value text-emerald">CUDA</div>
            <div className="stat-label">GPU Accelerated Inference</div>
          </div>
        </div>
      </div>
    </section>
  );
}
