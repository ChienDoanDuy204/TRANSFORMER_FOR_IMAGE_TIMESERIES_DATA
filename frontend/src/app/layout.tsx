import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ViT Image Captioning | Vision Transformer & Deep Learning Studio",
  description: "End-to-End Vision-Language Transformer for Vietnamese Image Captioning built with PyTorch, ViT Encoder, Transformer Decoder, FastAPI, and Next.js.",
  keywords: ["Vision Transformer", "ViT", "Image Captioning", "Deep Learning", "PyTorch", "FastAPI", "Next.js", "UIT-OpenVIIC"],
  authors: [{ name: "Chien Doan Duy" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  );
}
