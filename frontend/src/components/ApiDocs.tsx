"use client";

import React, { useState } from "react";

export default function ApiDocs() {
  const [activeLang, setActiveLang] = useState<"curl" | "python" | "javascript">("curl");
  const [copied, setCopied] = useState(false);

  const snippets = {
    curl: `# 1. Health check
curl -X GET http://localhost:8000/health

# 2. Generate caption from an uploaded image file
curl -X POST http://localhost:8000/api/generate-caption \\
  -F "file=@sample.jpg" \\
  -F "max_length=35" \\
  -F "search_mode=greedy" \\
  -F "clean_caption=true"

# 3. Generate caption using a sample image ID
curl -X POST http://localhost:8000/api/generate-caption \\
  -F "sample_id=00000000001.jpg" \\
  -F "max_length=35"`,

    python: `import requests

API_URL = "http://localhost:8000/api/generate-caption"

# Option A: Upload local image file
with open("test.jpg", "rb") as f:
    response = requests.post(
        API_URL,
        files={"file": f},
        data={
            "max_length": 35,
            "search_mode": "greedy",
            "clean_caption": "true"
        }
    )

print("Generated Caption:", response.json()["caption"])

# Option B: Use dataset sample ID
response = requests.post(
    API_URL,
    data={"sample_id": "00000000001.jpg", "max_length": 35}
)
print("Result:", response.json())`,

    javascript: `// Generate caption from Next.js or browser frontend
const formData = new FormData();
formData.append("file", imageFile); // File object from input
formData.append("max_length", "35");
formData.append("search_mode", "greedy");
formData.append("clean_caption", "true");

const response = await fetch("http://localhost:8000/api/generate-caption", {
  method: "POST",
  body: formData,
});

const data = await response.json();
console.log("Caption:", data.caption);
console.log("Latency:", data.latency_ms, "ms");`
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(snippets[activeLang]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="api" className="section" style={{ background: "rgba(13, 17, 26, 0.5)" }}>
      <div className="container">
        <div className="section-header">
          <div className="badge">Developer Integration</div>
          <h2 className="section-title">REST API Reference</h2>
          <p className="section-subtitle">
            Integrate the PyTorch Vision Transformer into external web services, mobile apps, or batch pipelines.
          </p>
        </div>

        {/* Code Block Container */}
        <div className="glass-card" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.75rem" }}>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                type="button"
                onClick={() => setActiveLang("curl")}
                className={`btn btn-sm ${activeLang === "curl" ? "btn-primary" : "btn-secondary"}`}
              >
                cURL
              </button>
              <button
                type="button"
                onClick={() => setActiveLang("python")}
                className={`btn btn-sm ${activeLang === "python" ? "btn-primary" : "btn-secondary"}`}
              >
                Python Requests
              </button>
              <button
                type="button"
                onClick={() => setActiveLang("javascript")}
                className={`btn btn-sm ${activeLang === "javascript" ? "btn-primary" : "btn-secondary"}`}
              >
                JavaScript / Next.js
              </button>
            </div>

            <button
              type="button"
              onClick={handleCopy}
              className="btn btn-secondary btn-sm"
            >
              {copied ? "✓ Copied!" : "Copy Snippet"}
            </button>
          </div>

          <pre style={{
            background: "#05070b",
            padding: "1.25rem",
            borderRadius: "var(--radius-md)",
            overflowX: "auto",
            fontSize: "0.85rem",
            lineHeight: 1.6,
            color: "#e2e8f0",
            border: "1px solid var(--border-subtle)",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
          }}>
            <code>{snippets[activeLang]}</code>
          </pre>

          {/* Endpoint Reference Table */}
          <div style={{ marginTop: "2rem" }}>
            <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.75rem" }}>
              Core API Endpoints
            </h4>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              <div style={{ display: "grid", gridTemplateColumns: "80px 240px 1fr", gap: "1rem", padding: "0.6rem 0.75rem", background: "rgba(255, 255, 255, 0.02)", borderRadius: "var(--radius-sm)", fontSize: "0.85rem", alignItems: "center" }}>
                <span className="mono text-emerald" style={{ fontWeight: 700 }}>GET</span>
                <span className="mono text-cyan">/health</span>
                <span style={{ color: "var(--text-secondary)" }}>System uptime, device (CUDA/CPU), model status</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "80px 240px 1fr", gap: "1rem", padding: "0.6rem 0.75rem", background: "rgba(255, 255, 255, 0.02)", borderRadius: "var(--radius-sm)", fontSize: "0.85rem", alignItems: "center" }}>
                <span className="mono text-emerald" style={{ fontWeight: 700 }}>GET</span>
                <span className="mono text-cyan">/api/model-info</span>
                <span style={{ color: "var(--text-secondary)" }}>Architecture hyperparameters, layers, parameter counts</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "80px 240px 1fr", gap: "1rem", padding: "0.6rem 0.75rem", background: "rgba(255, 255, 255, 0.02)", borderRadius: "var(--radius-sm)", fontSize: "0.85rem", alignItems: "center" }}>
                <span className="mono text-emerald" style={{ fontWeight: 700 }}>GET</span>
                <span className="mono text-cyan">/api/sample-images</span>
                <span style={{ color: "var(--text-secondary)" }}>List curated benchmark images with ground truth annotations</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "80px 240px 1fr", gap: "1rem", padding: "0.6rem 0.75rem", background: "rgba(255, 255, 255, 0.02)", borderRadius: "var(--radius-sm)", fontSize: "0.85rem", alignItems: "center" }}>
                <span className="mono text-indigo" style={{ fontWeight: 700 }}>POST</span>
                <span className="mono text-cyan">/api/generate-caption</span>
                <span style={{ color: "var(--text-secondary)" }}>Autoregressive caption generation (file or sample_id)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
