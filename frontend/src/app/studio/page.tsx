"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Hero from "@/components/Hero";
import Playground from "@/components/Playground";
import ArchitectureViewer from "@/components/ArchitectureViewer";
import DatasetMetrics from "@/components/DatasetMetrics";
import ApiDocs from "@/components/ApiDocs";
import Footer from "@/components/Footer";

interface UserSession {
  username: string;
  role: string;
  loginTime: string;
}

export default function StudioPage() {
  const router = useRouter();
  const [user, setUser] = useState<UserSession | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  useEffect(() => {
    // Check local storage for mock user authentication session
    try {
      const stored = localStorage.getItem("vit_user");
      if (stored) {
        setUser(JSON.parse(stored));
      } else {
        // Mock fallback if user directly navigated, or redirect
        const defaultUser: UserSession = {
          username: "Khách (Demo User)",
          role: "AI Researcher",
          loginTime: new Date().toISOString(),
        };
        setUser(defaultUser);
      }
    } catch {
      // ignore
    } finally {
      setIsCheckingAuth(false);
    }
  }, []);

  const handleLogout = () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem("vit_user");
    }
    router.push("/");
  };

  if (isCheckingAuth) {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#07090e" }}>
        <div className="badge badge-emerald">
          <span className="badge-pulse"></span>
          <span>Đang tải không gian làm việc AI Studio...</span>
        </div>
      </div>
    );
  }

  return (
    <main>
      {/* Studio Header Nav */}
      <header className="studio-header">
        <div className="container" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            <a href="#" className="nav-brand" style={{ gap: "0.5rem" }}>
              <div className="nav-logo-icon" style={{ width: 32, height: 32 }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                  <circle cx="12" cy="12" r="3" fill="#ffffff" />
                </svg>
              </div>
              <span style={{ fontSize: "1.05rem" }}>ViT Studio</span>
            </a>

            <ul className="nav-links" style={{ gap: "1.25rem" }}>
              <li><a href="#playground" className="nav-link">Playground</a></li>
              <li><a href="#architecture" className="nav-link">Kiến trúc</a></li>
              <li><a href="#dataset" className="nav-link">Dữ liệu &amp; BLEU</a></li>
              <li><a href="#api" className="nav-link">API Docs</a></li>
            </ul>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.85rem" }}>
            {/* User Profile Chip */}
            <div className="studio-user-chip">
              <div className="user-avatar">
                {user?.username?.charAt(0).toUpperCase() || "U"}
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: "0.825rem", color: "#ffffff" }}>
                  {user?.username || "Chuyên viên AI"}
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--accent-emerald)" }}>
                  ● Đã đăng nhập
                </div>
              </div>
            </div>

            {/* Logout Button */}
            <button
              type="button"
              onClick={handleLogout}
              className="btn btn-secondary btn-sm"
              style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}
              title="Đăng xuất và quay lại Landing Page"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
              <span>Đăng xuất</span>
            </button>
          </div>
        </div>
      </header>

      {/* Hero Welcome Banner */}
      <Hero />

      {/* Interactive AI Studio Workspace */}
      <Playground />

      {/* ViT & Decoder Pipeline Explorer */}
      <ArchitectureViewer />

      {/* UIT-OpenVIIC Dataset & Evaluation Metrics */}
      <DatasetMetrics />

      {/* API Reference */}
      <ApiDocs />

      {/* Footer */}
      <Footer />
    </main>
  );
}
