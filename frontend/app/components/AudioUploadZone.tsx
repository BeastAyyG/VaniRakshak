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
      if (file.type === "audio/wav" || /\.wav$/i.test(file.name)) {
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
        className={`relative flex flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center transition-colors cursor-pointer ${
          isDragging
            ? "border-white bg-white/[0.04]"
            : "border-white/[0.14] bg-[#080808] hover:border-white/[0.3] hover:bg-[#0f0f0f]"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/wav,.wav"
          className="hidden"
          onChange={handleFileChange}
          disabled={isAnalyzing}
        />

        {isAnalyzing ? (
          <div className="flex flex-col items-center py-3">
            <Loader2 className="h-8 w-8 animate-spin text-white" />
            <div className="mt-3 font-mono text-xs text-white">
              Evaluating sliding windows...
            </div>
            <p className="mt-1 text-xs text-[#888888]">
              Extracting 768-d WavLM acoustic representations and ECAPA-TDNN speaker similarity
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="flex h-11 w-11 items-center justify-center rounded-full border border-white/[0.08] bg-[#171717] text-white">
              <FileAudio className="h-5 w-5" />
            </div>

            <div className="mt-3 font-medium text-sm text-white">
              {currentFile ? currentFile.name : selectedFileName || "Select or drop audio file here."}
            </div>

            <p className="mt-1 text-xs text-[#888888]">
              WAV up to 50 MB · Standardized in volatile RAM to 16 kHz mono
            </p>

            <div className="mt-3 flex flex-wrap items-center justify-center gap-1.5 text-[10.5px] font-mono text-[#888888]">
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">2.0s windows</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">0.5s hop</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">768-d WavLM</span>
              <span className="rounded border border-white/[0.06] bg-[#121212] px-2 py-0.5">192-d ECAPA</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
