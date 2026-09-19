"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

interface SampleImage {
  id: string;
  filename: string;
  url: string;
  size_bytes: number;
  references: string[];
}

interface TokenDetail {
  step: number;
  id: number;
  token: string;
  confidence: number;
}

interface CaptionResponse {
  caption: string;
  raw_caption: string;
  tokens: string[];
  token_details: TokenDetail[];
  token_count: number;
  avg_confidence: number;
  latency_ms: number;
  search_mode: string;
  temperature: number;
  device: string;
  source: string;
  ground_truth?: string[] | null;
}

export default function Playground() {
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Settings
  const [showPatchGrid, setShowPatchGrid] = useState(false);
  const [maxLength, setMaxLength] = useState(35);
  const [searchMode, setSearchMode] = useState<"greedy" | "sampling">("greedy");
  const [temperature, setTemperature] = useState(0.7);
  const [cleanCaption, setCleanCaption] = useState(true);

  // Status & Output
  const [isGenerating, setIsGenerating] = useState(false);
  const [thinkingStep, setThinkingStep] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState<string>("");
  const [revealedTokens, setRevealedTokens] = useState<number>(0);
  const [result, setResult] = useState<CaptionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Cleanup audio playback on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  // Fetch sample images from backend on mount
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/sample-images`);
        if (res.ok) {
          const data = await res.json();
          setSamples(data.samples || []);
          if (data.samples && data.samples.length > 0) {
            selectSample(data.samples[0]);
          }
        }
      } catch (err) {
        console.warn("Could not fetch sample images:", err);
      }
    };
    fetchSamples();
  }, [apiUrl]);

  const selectSample = (sample: SampleImage) => {
    setSelectedImage(`${apiUrl}${sample.url}`);
    setSelectedFile(null);
    setSampleId(sample.id);
    setResult(null);
    setStreamedText("");
    setError(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setSelectedImage(URL.createObjectURL(file));
      setSampleId(null);
      setResult(null);
      setStreamedText("");
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setSelectedImage(URL.createObjectURL(file));
      setSampleId(null);
      setResult(null);
      setStreamedText("");
      setError(null);
    }
  };

  // Generate Caption API Call with Thinking & Word-by-Word Streaming Animation
  const handleGenerate = async () => {
    if (!selectedImage) {
      setError("Vui lòng chọn hoặc tải lên một hình ảnh trước khi sinh chú thích.");
      return;
    }

    if (timerRef.current) clearInterval(timerRef.current);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);

    setIsGenerating(true);
    setIsStreaming(false);
    setStreamedText("");
    setRevealedTokens(0);
    setError(null);
    setResult(null);

    setThinkingStep("🧠 ViT Encoder: Đang chia ảnh thành 196 Visual Patches 16×16...");

    // Simulated step transition for realism
    const stepTimer1 = setTimeout(() => {
      setThinkingStep("⚡ Multi-Head Cross-Attention: Đang đối chiếu đặc trưng ảnh với từ điển 10,000 từ...");
    }, 280);

    const stepTimer2 = setTimeout(() => {
      setThinkingStep("✍️ Causal Decoder: Đang dự đoán phân phối xác suất và sinh từng token tự hồi quy...");
    }, 550);

    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append("file", selectedFile);
      } else if (sampleId) {
        formData.append("sample_id", sampleId);
      }

      formData.append("max_length", maxLength.toString());
      formData.append("search_mode", searchMode);
      formData.append("temperature", temperature.toString());
      formData.append("clean_caption", cleanCaption ? "true" : "false");

      const response = await fetch(`${apiUrl}/api/generate-caption`, {
        method: "POST",
        body: formData,
      });

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned error ${response.status}`);
      }

      const data: CaptionResponse = await response.json();
      setResult(data);
      setIsGenerating(false);

      // Start Thinking Word-by-Word Streaming Animation
      setIsStreaming(true);
      const fullCaption = data.caption || "";
      const words = fullCaption.split(" ");
      let currentWordIndex = 0;

      timerRef.current = setInterval(() => {
        currentWordIndex++;
        setStreamedText(words.slice(0, currentWordIndex).join(" "));
        setRevealedTokens(Math.min(currentWordIndex, data.token_details?.length || currentWordIndex));

        if (currentWordIndex >= words.length) {
          if (timerRef.current) clearInterval(timerRef.current);
          setIsStreaming(false);
          setStreamedText(fullCaption);
          setRevealedTokens(data.token_details?.length || words.length);
        }
      }, 75); // 75ms per word for natural autoregressive pacing

    } catch (err: unknown) {
      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      const msg = err instanceof Error ? err.message : "Đã xảy ra lỗi khi kết nối tới máy chủ AI.";
      setError(msg);
      setIsGenerating(false);
      setIsStreaming(false);
    }
  };

  // High-Fidelity Natural Vietnamese Female Voice TTS
  const handleSpeak = useCallback(() => {
    const textToSpeak = streamedText || result?.caption;
    if (!textToSpeak) return;

    // Toggle stop if already speaking
    if (isSpeaking) {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      setIsSpeaking(false);
      return;
    }

    setIsSpeaking(true);

    const encoded = encodeURIComponent(textToSpeak.trim());
    const backendTtsUrl = `${apiUrl}/api/tts?text=${encoded}`;
    const directGoogleTtsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encoded}&tl=vi&client=tw-ob`;

    const playWithAudioElement = (url: string, onFail: () => void) => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onplay = () => setIsSpeaking(true);
      audio.onended = () => {
        setIsSpeaking(false);
        audioRef.current = null;
      };
      audio.onerror = () => {
        audioRef.current = null;
        onFail();
      };

      audio.play().catch(() => {
        audioRef.current = null;
        onFail();
      });
    };

    // Fallback: Browser Web Speech API with tuned natural female voice
    const playWithWebSpeech = () => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) {
        setIsSpeaking(false);
        return;
      }

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.lang = "vi-VN";
      utterance.rate = 0.92; // Natural, steady human pace (avoids stuttering/giật cục)
      utterance.pitch = 1.06; // Sweet, gentle female vocal pitch

      const voices = window.speechSynthesis.getVoices();
      // Search for female voices or Google Vietnamese voice
      const viFemaleVoice = voices.find(
        (v) =>
          (v.lang.includes("vi") || v.lang.includes("VN")) &&
          (v.name.toLowerCase().includes("hoaimy") ||
            v.name.toLowerCase().includes("linh") ||
            v.name.toLowerCase().includes("mai") ||
            v.name.toLowerCase().includes("female") ||
            v.name.toLowerCase().includes("google") ||
            v.name.toLowerCase().includes("natural"))
      ) || voices.find((v) => v.lang.includes("vi") || v.lang.includes("VN"));

      if (viFemaleVoice) {
        utterance.voice = viFemaleVoice;
      }

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => {
        setIsSpeaking(false);
      };
      utterance.onerror = () => {
        setIsSpeaking(false);
      };

      window.speechSynthesis.speak(utterance);
    };

    // Multi-tier Fallback: 1. Backend cached gTTS -> 2. Direct Google TTS -> 3. Tuned Web Speech API
    playWithAudioElement(backendTtsUrl, () => {
      playWithAudioElement(directGoogleTtsUrl, () => {
        playWithWebSpeech();
      });
    });
  }, [result, streamedText, isSpeaking, apiUrl]);

  const handleCopy = () => {
    const textToCopy = streamedText || result?.caption;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section id="playground" className="section">
      <div className="container">
        <div className="section-header">
          <div className="badge">Interactive AI Studio</div>
          <h2 className="section-title">Vision Transformer Playground</h2>
          <p className="section-subtitle">
            Tải ảnh tùy ý hoặc chọn ảnh từ tập benchmark UIT-OpenVIIC để theo dõi mô hình sinh chú thích tự hồi quy từng chữ một (Thinking Animation).
          </p>
        </div>

        <div className="playground-grid">
          {/* Left Column: Image Input & Visualization */}
          <div className="glass-card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Ảnh Đầu Vào (ViT Input)</h3>
              
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button
                  type="button"
                  onClick={() => setShowPatchGrid(!showPatchGrid)}
                  className={`btn btn-sm ${showPatchGrid ? "btn-primary" : "btn-secondary"}`}
                  title="Bật/Tắt hiển thị lưới 196 visual patches (16×16)"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <line x1="9" y1="3" x2="9" y2="21" />
                    <line x1="15" y1="3" x2="15" y2="21" />
                    <line x1="3" y1="9" x2="21" y2="9" />
                    <line x1="3" y1="15" x2="21" y2="15" />
                  </svg>
                  <span>Lưới 16×16 Patches</span>
                </button>

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-secondary btn-sm"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <span>Tải ảnh lên</span>
                </button>
              </div>
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: "none" }}
            />

            {/* Preview Box */}
            <div
              className={`image-box ${isDragOver ? "drag-over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              onClick={() => !selectedImage && fileInputRef.current?.click()}
            >
              {selectedImage ? (
                <div className="preview-container">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={selectedImage}
                    alt="Ảnh cho mô hình sinh chú thích"
                    className="preview-img"
                  />
                  {showPatchGrid && (
                    <div className="patch-overlay">
                      {Array.from({ length: 196 }).map((_, i) => (
                        <div key={i} className="patch-cell" title={`Visual Patch #${i + 1}`} />
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: "2rem" }}>
                  <div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>🖼️</div>
                  <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>
                    Kéo thả ảnh vào đây hoặc bấm để chọn tệp từ máy
                  </p>
                  <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                    Hỗ trợ PNG, JPG, JPEG, WEBP (Tự động chuẩn hóa 224×224 cho ViT)
                  </p>
                </div>
              )}
            </div>

            {/* Curated Sample Gallery */}
            <div className="sample-gallery">
              <div className="sample-gallery-title">
                <span>Bộ sưu tập ảnh mẫu chuẩn (UIT-OpenVIIC Benchmark)</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  {samples.length} ảnh có sẵn
                </span>
              </div>
              <div className="sample-strip">
                {samples.map((sample) => (
                  <div
                    key={sample.id}
                    className={`sample-thumb ${sampleId === sample.id ? "active" : ""}`}
                    onClick={() => selectSample(sample)}
                    title={sample.references[0] || sample.filename}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`${apiUrl}${sample.url}`} alt={sample.filename} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Generation Controls & Thinking Stream Output */}
          <div className="glass-card">
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "1.25rem" }}>
              Tùy Chọn Giải Mã Transformer (Decoder Settings)
            </h3>

            {/* Max Length Slider */}
            <div className="control-group">
              <div className="control-label">
                <span>Độ dài tối đa (Max Sequence Tokens)</span>
                <span className="mono text-cyan">{maxLength} tokens</span>
              </div>
              <input
                type="range"
                min={10}
                max={60}
                value={maxLength}
                onChange={(e) => setMaxLength(Number(e.target.value))}
                className="slider-input"
              />
            </div>

            {/* Decoding Strategy */}
            <div className="control-group">
              <div className="control-label">
                <span>Chiến Lược Giải Mã (Decoding Mode)</span>
                <span className="mono text-indigo">{searchMode === "greedy" ? "GREEDY SEARCH" : "TEMPERATURE SAMPLING"}</span>
              </div>
              <select
                value={searchMode}
                onChange={(e) => setSearchMode(e.target.value as "greedy" | "sampling")}
                className="select-input"
              >
                <option value="greedy">Greedy Search (Tham lam - Chọn token xác suất cao nhất)</option>
                <option value="sampling">Temperature Sampling (Lấy mẫu xác suất ngẫu nhiên)</option>
              </select>
            </div>

            {/* Temperature Slider */}
            {searchMode === "sampling" && (
              <div className="control-group">
                <div className="control-label">
                  <span>Hệ Số Nhiệt Độ (Temperature Scaling)</span>
                  <span className="mono text-cyan">{temperature}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={1.5}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="slider-input"
                />
              </div>
            )}

            {/* Clean Caption Toggle */}
            <div className="control-group" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div>
                <div style={{ fontSize: "0.875rem", fontWeight: 600 }}>Bộ Lọc Làm Đẹp Tiếng Việt</div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Tự động chuyển từ ghép (dấu gạch dưới), viết hoa và lọc token đặc biệt
                </div>
              </div>
              <input
                type="checkbox"
                checked={cleanCaption}
                onChange={(e) => setCleanCaption(e.target.checked)}
                style={{ width: "18px", height: "18px", accentColor: "var(--accent-indigo)" }}
              />
            </div>

            {/* Generate Action Button */}
            <button
              type="button"
              onClick={handleGenerate}
              disabled={Boolean(isGenerating || isStreaming || !selectedImage)}
              className="btn btn-primary"
              style={{ width: "100%", padding: "0.95rem", marginTop: "0.75rem" }}
            >
              {isGenerating || isStreaming ? (
                <>
                  <svg className="badge-pulse" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="10" strokeDasharray="30" strokeDashoffset="10" />
                  </svg>
                  <span>{isGenerating ? "Mô Hình Đang Tính Toán..." : "Đang Sinh Từng Chữ Một..."}</span>
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Sinh Chú Thích Tiếng Việt (Thinking Mode)</span>
                </>
              )}
            </button>

            {/* Real-time Thinking Step Notification */}
            {isGenerating && (
              <div className="thinking-box" style={{ marginTop: "1rem" }}>
                <span className="thinking-pulse-dot"></span>
                <span>{thinkingStep}</span>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div style={{ marginTop: "1rem", padding: "0.85rem", background: "rgba(244, 63, 94, 0.12)", border: "1px solid rgba(244, 63, 94, 0.3)", borderRadius: "var(--radius-md)", color: "var(--accent-rose)", fontSize: "0.875rem" }}>
                {error}
              </div>
            )}

            {/* Results Display with Thinking Stream */}
            {(result || isStreaming) && (
              <div className="result-card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div className="badge badge-emerald">
                    <span className="badge-pulse"></span>
                    <span>
                      {isStreaming ? "Đang giải mã từng chữ..." : `Suy Luận Thành Công (${result?.latency_ms}ms)`}
                    </span>
                  </div>

                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    <button
                      type="button"
                      onClick={handleSpeak}
                      disabled={isStreaming}
                      className={`btn ${isSpeaking ? "btn-primary" : "btn-secondary"} btn-sm`}
                      title="Phát âm tiếng Việt bằng giọng nữ tự nhiên"
                      style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}
                    >
                      {isSpeaking ? (
                        <>
                          <span className="badge-pulse" style={{ width: 8, height: 8, background: "#38bdf8" }}></span>
                          <span>Đang Đọc Giọng Nữ... (Dừng)</span>
                        </>
                      ) : (
                        <>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
                          </svg>
                          <span>Phát Âm (Giọng Nữ)</span>
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={handleCopy}
                      disabled={isStreaming}
                      className="btn btn-secondary btn-sm"
                      title="Sao chép chú thích"
                    >
                      {copied ? "✓ Đã Chép!" : "Sao Chép"}
                    </button>
                  </div>
                </div>

                {/* Word-by-Word Thinking Animation Caption Output */}
                <div className="caption-output" style={{ minHeight: "60px" }}>
                  &ldquo;{streamedText || result?.caption}
                  {isStreaming && <span className="typing-cursor">▋</span>}
                  &rdquo;
                </div>

                {/* Telemetry Bar */}
                {result && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", margin: "1rem 0", padding: "0.75rem", background: "rgba(0, 0, 0, 0.25)", borderRadius: "var(--radius-sm)", fontSize: "0.8rem" }}>
                    <div>
                      <span style={{ color: "var(--text-muted)", display: "block" }}>Độ trễ GPU</span>
                      <span className="mono text-cyan" style={{ fontWeight: 700 }}>{result.latency_ms} ms</span>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)", display: "block" }}>Số từ sinh ra</span>
                      <span className="mono" style={{ fontWeight: 700 }}>
                        {isStreaming ? revealedTokens : result.token_count}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)", display: "block" }}>Độ tin cậy TB</span>
                      <span className="mono text-emerald" style={{ fontWeight: 700 }}>
                        {(result.avg_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )}

                {/* Ground Truth Reference Comparison if available */}
                {result?.ground_truth && result.ground_truth.length > 0 && (
                  <div style={{ marginTop: "1rem", paddingTop: "0.75rem", borderTop: "1px solid var(--border-subtle)" }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)", marginBottom: "0.4rem" }}>
                      Chú Thích Tham Chiếu Gốc Của Con Người (UIT-OpenVIIC Ground Truth):
                    </div>
                    <ul style={{ listStyle: "disc", paddingLeft: "1.25rem", fontSize: "0.825rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
                      {result.ground_truth.slice(0, 3).map((ref, idx) => (
                        <li key={idx} style={{ fontStyle: "italic" }}>{ref}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Token Sequence Breakdown with Sequential Pop-in Animation */}
                {result?.token_details && result.token_details.length > 0 && (
                  <div style={{ marginTop: "1rem" }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.4rem" }}>
                      Chuỗi Token Giải Mã Tự Hồi Quy (Autoregressive Steps):
                    </div>
                    <div className="token-stream">
                      {result.token_details.slice(0, revealedTokens).map((td) => (
                        <div
                          key={td.step}
                          className="token-chip token-chip-emerge"
                          title={`Bước ${td.step} | ID: ${td.id} | Độ tin cậy: ${(td.confidence * 100).toFixed(1)}%`}
                        >
                          <span>{td.token}</span>
                          <span className="token-conf">{(td.confidence * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
