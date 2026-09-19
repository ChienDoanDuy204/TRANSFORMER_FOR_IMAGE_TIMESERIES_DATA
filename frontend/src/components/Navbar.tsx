"use client";

import React, { useEffect, useState } from "react";

interface HealthData {
  status: string;
  uptime_seconds: number;
  device: string;
  vocab_size: number;
  model_loaded: boolean;
}

export default function Navbar() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);
  const [latency, setLatency] = useState<number | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      const startTime = performance.now();
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/health`);
        if (res.ok) {
          const data: HealthData = await res.json();
          setHealth(data);
          setIsOnline(true);
          setLatency(Math.round(performance.now() - startTime));
        } else {
          setIsOnline(false);
        }
      } catch {
        setIsOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="navbar">
      <div className="container navbar-inner">
        <a href="#" className="nav-brand">
          <div className="nav-logo-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              <circle cx="12" cy="12" r="3" fill="#ffffff" />
            </svg>
          </div>
          <div>
            <span>ViT</span>
            <span className="text-gradient" style={{ marginLeft: "4px" }}>Captioner</span>
          </div>
        </a>

        <ul className="nav-links">
          <li><a href="#playground" className="nav-link">Studio</a></li>
          <li><a href="#architecture" className="nav-link">Architecture</a></li>
          <li><a href="#dataset" className="nav-link">Dataset & Metrics</a></li>
          <li><a href="#api" className="nav-link">API Docs</a></li>
        </ul>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          {isOnline ? (
            <div className="badge badge-emerald">
              <span className="badge-pulse"></span>
              <span>FastAPI {health?.device ? `(${health.device.toUpperCase()})` : "Online"}</span>
              {latency && <span style={{ opacity: 0.7, fontSize: "0.75rem" }}>{latency}ms</span>}
            </div>
          ) : (
            <div className="badge" style={{ background: "rgba(244, 63, 94, 0.1)", borderColor: "rgba(244, 63, 94, 0.3)", color: "var(--accent-rose)" }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent-rose)" }}></span>
              <span>Backend Offline</span>
            </div>
          )}

          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary btn-sm"
          >
            Swagger UI
          </a>
        </div>
      </div>
    </nav>
  );
}
