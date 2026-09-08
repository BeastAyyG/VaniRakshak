"use client";

import React, { useRef, useState } from "react";
import { FileAudio, Loader2, ShieldCheck } from "lucide-react";

interface AudioUploadZoneProps {
  onFileSelected: (file: File) => void;
  isAnalyzing: boolean;
  selectedFileName?: string | null;
}

export default function AudioUploadZone({
  onFileSelected,
  isAnalyzing,
  selectedFileName,
}: AudioUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("audio/") || /\.(wav|mp3|m4a|ogg|flac|aac)$/i.test(file.name)) {
        setCurrentFile(file);
        onFileSelected(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setCurrentFile(file);
      onFileSelected(file);
    }
  };

  const triggerPicker = () => {
    if (!isAnalyzing) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <span className="font-mono text-xs uppercase tracking-wider text-[#a1a1a1]">
            AUDIO INGESTION PIPELINE
          </span>
          <h3 className="mt-1 text-lg font-semibold tracking-[-0.6px] text-white">
            Custom audio file evaluation.
          </h3>
        </div>
        <div className="flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-[#171717] px-3 py-1 text-[11px] font-mono text-[#a1a1a1]">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
          <span>DPDP Act §8(7) compliant</span>
        </div>
      </div>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerPicker}
        className={`relative flex flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center transition-all cursor-pointer ${
          isDragging
            ? "border-emerald-400 bg-emerald-500/[0.05] shadow-[0_0_25px_rgba(16,185,129,0.15)]"
            : "border-white/[0.14] bg-[#080808] hover:border-white/[0.3] hover:bg-[#0f0f0f]"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
          className="hidden"
          onChange={handleFileChange}
          disabled={isAnalyzing}
        />

        {isAnalyzing ? (
          <div className="flex flex-col items-center py-4">
            {/* Wispr Flow-style animated pulsing soundwave bars */}
            <div className="flex items-center gap-1.5 h-10 mb-3">
              {[40, 75, 100, 60, 90, 45, 80, 55, 30].map((height, i) => (
                <div
                  key={i}
                  className="w-1.5 rounded-full bg-gradient-to-t from-emerald-500 to-teal-300 animate-pulse"
                  style={{
                    height: `${height}%`,
                    animationDelay: `${i * 120}ms`,
                    animationDuration: "900ms",
                  }}
                />
              ))}
            </div>
            <div className="font-mono text-xs font-medium text-white flex items-center gap-2">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-400" />
              <span>Analyzing forensic sliding windows...</span>
            </div>
            <p className="mt-1 text-xs text-[#888888]">
              Extracting WavLM acoustic representations, vocoder boundaries & speaker identity
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full border border-white/[0.08] bg-[#171717] text-white shadow-inner group-hover:scale-105 transition-transform">
              <FileAudio className="h-5 w-5 text-neutral-300" />
            </div>

            <div className="mt-3 font-medium text-sm text-white">
              {currentFile ? currentFile.name : selectedFileName || "Select or drop audio file here."}
            </div>

            <p className="mt-1 text-xs text-[#888888]">
              WAV, MP3, FLAC, M4A up to 50 MB · In-memory 16 kHz mono standardization
            </p>

            <div className="mt-3.5 flex flex-wrap items-center justify-center gap-1.5 text-[10.5px] font-mono text-[#888888]">
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">16 kHz PCM</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">4.04s windows</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">1.0s hop</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">Biometric Identity</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">Zero-Retention</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
