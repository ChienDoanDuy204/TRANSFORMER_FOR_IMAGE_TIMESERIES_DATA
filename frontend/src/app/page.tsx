"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

interface MosaicImage {
  id: string;
  filename: string;
  src: string;
  caption: string;
}

const MOSAIC_ITEMS: MosaicImage[] = [
  {
    id: "0",
    filename: "00000000000.png",
    src: "/samples/00000000000.png",
    caption: "Một chiếc khiên màu xanh dương với viền ngoài màu trắng bạc",
  },
  {
    id: "1",
    filename: "00000000001.jpg",
    src: "/samples/00000000001.jpg",
    caption: "Ba ngôi nhà cạnh nhau có số tầng không ít hơn một",
  },
  {
    id: "2",
    filename: "00000000002.jpg",
    src: "/samples/00000000002.jpg",
    caption: "Một cánh đồng lúa chín bên cạnh một dòng sông",
  },
  {
    id: "3",
    filename: "00000000003.jpg",
    src: "/samples/00000000003.jpg",
    caption: "Người phụ nữ mặc đồ dân tộc đang cõng con trên vai",
  },
  {
    id: "4",
    filename: "00000000004.jpg",
    src: "/samples/00000000004.jpg",
    caption: "Cô gái đeo túi xách ngắm lồng đèn đầy màu sắc rực rỡ",
  },
  {
    id: "5",
    filename: "00000000005.jpg",
    src: "/samples/00000000005.jpg",
    caption: "Phiên chợ với rất nhiều người dân trang phục đa dạng mua bán",
  },
  {
    id: "6",
    filename: "00000000006.jpg",
    src: "/samples/00000000006.jpg",
    caption: "Bốn người đang ăn trong một quán ăn bên đường",
  },
  {
    id: "7",
    filename: "00000000007.jpg",
    src: "/samples/00000000007.jpg",
    caption: "Có nhiều đứa trẻ đang chơi đùa trước sân một ngôi nhà lớn",
  },
  {
    id: "8",
    filename: "00000000008.jpg",
    src: "/samples/00000000008.jpg",
    caption: "Các nghệ sĩ đang biểu diễn trên đường phố với trang phục dân tộc",
  },
  {
    id: "9",
    filename: "00000000009.jpg",
    src: "/samples/00000000009.jpg",
    caption: "Cô gái đang tạo dáng ở công viên với hai bên là cánh đồng hoa",
  },
  {
    id: "10",
    filename: "00000000001.jpg",
    src: "/samples/00000000001.jpg",
    caption: "Chiếc xe tải đậu bên hông căn nhà ở giữa có nhiều tầng",
  },
  {
    id: "11",
    filename: "00000000002.jpg",
    src: "/samples/00000000002.jpg",
    caption: "Ba chiếc xuồng đang đi trên sông cạnh đồng lúa vàng óng",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: -500, y: -500 });
  const [isLensVisible, setIsLensVisible] = useState<boolean>(false);
  const [activeImage, setActiveImage] = useState<MosaicImage | null>(MOSAIC_ITEMS[1]);
  const [showLoginModal, setShowLoginModal] = useState<boolean>(false);
  const [username, setUsername] = useState<string>("admin");
  const [password, setPassword] = useState<string>("123456");
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<Array<HTMLDivElement | null>>([]);

  // Check backend health
  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.status === "healthy") {
          setBackendOnline(true);
        }
      })
      .catch(() => {});
  }, []);

  // Track mouse and detect which image is under or nearest to the magnifying lens
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setMousePos({ x, y });
    setIsLensVisible(true);

    containerRef.current.style.setProperty("--mouse-x", `${x}px`);
    containerRef.current.style.setProperty("--mouse-y", `${y}px`);

    // Find nearest cell
    let closestItem: MosaicImage | null = null;
    let minDistance = Infinity;

    cellRefs.current.forEach((cell, idx) => {
      if (!cell) return;
      const cellRect = cell.getBoundingClientRect();
      const cellCenterX = cellRect.left - rect.left + cellRect.width / 2;
      const cellCenterY = cellRect.top - rect.top + cellRect.height / 2;
      const dist = Math.hypot(x - cellCenterX, y - cellCenterY);

      if (dist < minDistance) {
        minDistance = dist;
        closestItem = MOSAIC_ITEMS[idx] || null;
      }
    });

    if (closestItem && minDistance < 280) {
      setActiveImage(closestItem);
    }
  };

  const handleMouseLeave = () => {
    setIsLensVisible(false);
  };

  // Mock Login Handler
  const handleLogin = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const userSession = {
      username: username.trim() || "Chuyên viên AI",
      role: "AI Researcher",
      loginTime: new Date().toISOString(),
    };
    if (typeof window !== "undefined") {
      localStorage.setItem("vit_user", JSON.stringify(userSession));
    }
    router.push("/studio");
  };

  return (
    <div
      ref={containerRef}
      className="landing-container"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Dimmed Background Mosaic Grid of Images */}
      <div className="landing-mosaic-grid">
        {MOSAIC_ITEMS.map((item, idx) => {
          const isActive = activeImage?.id === item.id && isLensVisible;
          return (
            <div
              key={`${item.id}-${idx}`}
              ref={(el) => { cellRefs.current[idx] = el; }}
              className={`mosaic-cell ${isActive ? "active-lens" : ""}`}
              onClick={() => setShowLoginModal(true)}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={item.src} alt={item.caption} />
              
              {/* Caption Overlay over the image card */}
              <div className="mosaic-caption-overlay">
                <div style={{ color: "var(--accent-cyan)", fontSize: "0.7rem", fontWeight: 700, marginBottom: "0.2rem" }}>
                  ✦ UIT-OpenVIIC Annotation:
                </div>
                <div>&ldquo;{item.caption}&rdquo;</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Atmospheric Dark Vignette Overlay */}
      <div className="landing-overlay" style={{ pointerEvents: "none" }} />

      {/* Authentic High-Tech Magnifying Glass Tool Following Mouse */}
      <div
        className="magnifier-tool-wrapper"
        style={{
          opacity: isLensVisible ? 1 : 0,
        }}
      >
        {/* Optical Glass Lens Element with dynamic contrast & refraction */}
        <div className="magnifier-glass-element" />

        {/* Realistic SVG Magnifying Glass (Bezel, Grip Handle, Reticle & Optical Glare) */}
        <svg
          className="magnifier-svg-graphic"
          viewBox="0 0 320 320"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            {/* Multi-tone chrome/titanium metallic bezel gradient */}
            <linearGradient id="metallic-rim" x1="20" y1="20" x2="180" y2="180" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#f8fafc" />
              <stop offset="20%" stopColor="#94a3b8" />
              <stop offset="45%" stopColor="#334155" />
              <stop offset="65%" stopColor="#64748b" />
              <stop offset="85%" stopColor="#cbd5e1" />
              <stop offset="100%" stopColor="#0f172a" />
            </linearGradient>

            {/* Inner rim glowing accent */}
            <linearGradient id="cyan-rim-accent" x1="0" y1="0" x2="200" y2="200" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="50%" stopColor="#818cf8" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>

            {/* Tactile ergonomic handle gradient across handle shaft */}
            <linearGradient id="handle-body" x1="0" y1="-12" x2="0" y2="12" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#334155" />
              <stop offset="25%" stopColor="#64748b" />
              <stop offset="50%" stopColor="#1e293b" />
              <stop offset="75%" stopColor="#0f172a" />
              <stop offset="100%" stopColor="#020617" />
            </linearGradient>

            {/* Polished chrome trim for ferrule rings & pommel */}
            <linearGradient id="chrome-accent" x1="0" y1="-12" x2="0" y2="12" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor="#ffffff" />
              <stop offset="35%" stopColor="#38bdf8" />
              <stop offset="70%" stopColor="#1e293b" />
              <stop offset="100%" stopColor="#e2e8f0" />
            </linearGradient>

            {/* Specular curved optical glare */}
            <linearGradient id="lens-glare-grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
              <stop offset="60%" stopColor="#38bdf8" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* 1. HANDLE ASSEMBLY: Rotated 45 degrees extending to bottom-right */}
          <g transform="translate(100, 100) rotate(45)">
            {/* Neck Ferrule Bracket connecting to outer rim at x=74..98 */}
            <path
              d="M 74,-11 C 80,-9 84,-6 88,-7 L 98,-7 L 98,7 L 88,7 C 84,6 80,9 74,11 Z"
              fill="url(#metallic-rim)"
              stroke="#0f172a"
              strokeWidth="1"
            />
            {/* Upper Chrome Accent Collar Ring */}
            <rect x="96" y="-8.5" width="6" height="17" rx="2" fill="url(#chrome-accent)" stroke="#0284c7" strokeWidth="0.8" />

            {/* Ergonomic Textured Handle Shaft */}
            <path
              d="M 102,-8.5 L 206,-10 Q 212,-10 212,-7.5 L 212,7.5 Q 212,10 206,10 L 102,8.5 Z"
              fill="url(#handle-body)"
              stroke="#334155"
              strokeWidth="1.2"
            />

            {/* Grip Knurling Bands & Cyber Grooves */}
            {[116, 126, 136, 146, 156, 166, 176, 186, 196].map((gx, i) => (
              <g key={`grip-${gx}`}>
                <line
                  x1={gx}
                  y1={-8.8 - (gx > 150 ? 0.8 : 0)}
                  x2={gx}
                  y2={8.8 + (gx > 150 ? 0.8 : 0)}
                  stroke={i % 2 === 0 ? "rgba(56, 189, 248, 0.75)" : "rgba(255, 255, 255, 0.25)"}
                  strokeWidth={i % 2 === 0 ? "2" : "1.2"}
                />
              </g>
            ))}

            {/* Lower Chrome Collar Ring */}
            <rect x="206" y="-9.5" width="5" height="19" rx="1.5" fill="url(#chrome-accent)" stroke="#0284c7" strokeWidth="0.8" />

            {/* Weighted Ergonomic Pommel End-Cap */}
            <path
              d="M 211,-8 Q 226,-10 228,0 Q 226,10 211,8 Z"
              fill="url(#metallic-rim)"
              stroke="#38bdf8"
              strokeWidth="1"
            />
            <circle cx="223" cy="0" r="3.2" fill="#38bdf8" />
            <circle cx="223" cy="0" r="1.5" fill="#ffffff" />
          </g>

          {/* 2. CIRCULAR LENS HEAD (Centered exactly at 100, 100) */}
          <g>
            {/* Outer Heavy Beveled Metallic Ring (Radius 78, stroke 8) */}
            <circle
              cx="100"
              cy="100"
              r="78"
              fill="none"
              stroke="url(#metallic-rim)"
              strokeWidth="8"
            />

            {/* Inner Precision Cyan LED Accent Ring */}
            <circle
              cx="100"
              cy="100"
              r="73.5"
              fill="none"
              stroke="url(#cyan-rim-accent)"
              strokeWidth="1.8"
            />

            {/* Optical Measurement Ticks around the inner rim */}
            {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((deg) => {
              const rad = (deg * Math.PI) / 180;
              const x1 = 100 + 72 * Math.cos(rad);
              const y1 = 100 + 72 * Math.sin(rad);
              const len = deg % 90 === 0 ? 5.5 : 3.5;
              const x2 = 100 + (72 - len) * Math.cos(rad);
              const y2 = 100 + (72 - len) * Math.sin(rad);
              return (
                <line
                  key={`tick-${deg}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={deg % 90 === 0 ? "rgba(56, 189, 248, 0.9)" : "rgba(148, 163, 184, 0.5)"}
                  strokeWidth={deg % 90 === 0 ? "1.5" : "1"}
                />
              );
            })}

            {/* Tactical Center Reticle / Precision Crosshairs */}
            <circle
              cx="100"
              cy="100"
              r="16"
              fill="none"
              stroke="rgba(56, 189, 248, 0.45)"
              strokeWidth="1"
              strokeDasharray="3,3"
            />
            <line x1="91" y1="100" x2="109" y2="100" stroke="#38bdf8" strokeWidth="1.4" />
            <line x1="100" y1="91" x2="100" y2="109" stroke="#38bdf8" strokeWidth="1.4" />
            <circle cx="100" cy="100" r="2.2" fill="#38bdf8" />
            <circle cx="100" cy="100" r="0.8" fill="#ffffff" />

            {/* Specular Optical Reflection Glare Arcs (Top-left quadrant crescent) */}
            <path
              d="M 52,65 A 66,66 0 0,1 96,34"
              fill="none"
              stroke="url(#lens-glare-grad)"
              strokeWidth="4.5"
              strokeLinecap="round"
            />
            <path
              d="M 44,82 A 66,66 0 0,1 58,56"
              fill="none"
              stroke="rgba(255, 255, 255, 0.65)"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
          </g>
        </svg>

        {/* Top Identification Badge */}
        <div className="magnifier-top-label">
          <span className="badge-pulse" style={{ width: 6, height: 6 }}></span>
          <span>🔍 KÍNH LÚP SOI ẢNH &amp; CHÚ THÍCH</span>
        </div>

        {/* Dynamic Caption HUD Floating Below the Magnifying Glass Lens */}
        {activeImage && (
          <div className="lens-caption-hud">
            <div className="lens-caption-hud-title">
              <span className="badge-pulse" style={{ width: 6, height: 6 }}></span>
              <span>Chú Thích UIT-OpenVIIC Nhận Diện Được</span>
            </div>
            <div className="lens-caption-hud-text">
              &ldquo;{activeImage.caption}&rdquo;
            </div>
          </div>
        )}
      </div>

      {/* Interactive UI Overlay (Navbar & Center CTA) */}
      <div className="landing-interactive-ui">
        {/* Top Navbar */}
        <div className="landing-nav">
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", fontWeight: 800, fontSize: "1.2rem" }}>
            <div className="nav-logo-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                <circle cx="12" cy="12" r="3" fill="#ffffff" />
              </svg>
            </div>
            <span>ViT</span>
            <span className="text-gradient">Captioner</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            {backendOnline ? (
              <div className="badge badge-emerald">
                <span className="badge-pulse"></span>
                <span>FastAPI AI Server Online</span>
              </div>
            ) : (
              <div className="badge" style={{ color: "var(--accent-amber)", borderColor: "rgba(245, 158, 11, 0.3)" }}>
                <span>Backend Port 8000</span>
              </div>
            )}

            <button
              type="button"
              onClick={() => setShowLoginModal(true)}
              className="btn btn-secondary btn-sm"
              style={{ padding: "0.5rem 1.1rem" }}
            >
              Đăng nhập
            </button>
          </div>
        </div>

        {/* Center Prompt & Call to Action */}
        <div className="landing-center-content">
          <div className="hint-pill">
            <span>🔍</span>
            <span>Di chuyển chuột như kính lúp để soi ảnh chìm và xem chú thích tiếng Việt tương ứng</span>
          </div>

          <h1
            style={{
              fontSize: "clamp(2rem, 4.5vw, 3.5rem)",
              fontWeight: 900,
              letterSpacing: "-0.03em",
              lineHeight: 1.15,
              marginBottom: "1rem",
              textShadow: "0 4px 24px rgba(0, 0, 0, 0.9)",
            }}
          >
            Vision Transformer &amp;{" "}
            <span className="text-gradient">Vietnamese Image Captioning</span>
          </h1>

          <p
            style={{
              fontSize: "1.15rem",
              color: "#cbd5e1",
              maxWidth: "680px",
              margin: "0 auto 2rem auto",
              lineHeight: 1.6,
              textShadow: "0 2px 12px rgba(0, 0, 0, 0.8)",
            }}
          >
            Khám phá kiến trúc đa phương thức kết hợp phân mảnh ảnh ViT 16×16 và bộ giải mã Causal Decoder sinh chú thích tự động trên tập ngữ liệu UIT-OpenVIIC.
          </p>

          <div className="landing-cta-container">
            <button
              type="button"
              onClick={() => setShowLoginModal(true)}
              className="btn btn-primary"
              style={{
                padding: "0.95rem 2.25rem",
                fontSize: "1.05rem",
                borderRadius: "var(--radius-full)",
                boxShadow: "0 0 35px rgba(99, 102, 241, 0.6)",
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
                <polyline points="10 17 15 12 10 7" />
                <line x1="15" y1="12" x2="3" y2="12" />
              </svg>
              <span>Đăng Nhập Để Vào AI Studio</span>
            </button>
          </div>
        </div>

        {/* Footer info bar */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.8rem", color: "var(--text-muted)", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>196 ViT Patches • 25.59M Parameters • 10,000 Vocab • 3-Layer Decoder • PyTorch 2.14 / CUDA</div>
          <div>© {new Date().getFullYear()} UIT-OpenVIIC Vision-Language Transformer</div>
        </div>
      </div>

      {/* Mock Login Modal */}
      {showLoginModal && (
        <div className="modal-backdrop" onClick={() => setShowLoginModal(false)}>
          <div className="login-modal" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="modal-close-btn"
              onClick={() => setShowLoginModal(false)}
              title="Đóng"
            >
              ✕
            </button>

            <div style={{ textAlign: "center", marginBottom: "1.75rem" }}>
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: "var(--radius-md)",
                  background: "var(--gradient-brand)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 1rem auto",
                  boxShadow: "var(--shadow-glow)",
                }}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="2.5">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
              </div>

              <h3 style={{ fontSize: "1.35rem", fontWeight: 800 }}>Đăng Nhập Studio</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.3rem" }}>
                Đăng nhập để mở khóa trang trải nghiệm Vision Transformer
              </p>
            </div>

            <form onSubmit={handleLogin}>
              <div className="form-group">
                <label className="form-label">Tài khoản / Email</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Nhập tên đăng nhập hoặc email..."
                  className="form-input"
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Mật khẩu</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="form-input"
                  required
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary"
                style={{ width: "100%", padding: "0.85rem", marginTop: "0.5rem" }}
              >
                Đăng Nhập
              </button>

              <div style={{ textAlign: "center", margin: "1rem 0", color: "var(--text-muted)", fontSize: "0.8rem" }}>
                ── HOẶC ──
              </div>

              <button
                type="button"
                onClick={() => handleLogin()}
                className="btn btn-secondary"
                style={{
                  width: "100%",
                  padding: "0.8rem",
                  borderColor: "var(--accent-cyan)",
                  color: "var(--accent-cyan)",
                }}
              >
                <span>🚀 Đăng Nhập Nhanh (Demo)</span>
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
