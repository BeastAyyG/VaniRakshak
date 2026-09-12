"use client";

import React from "react";
import { X, Square, Shield } from "lucide-react";
import VoiceVisualizer from "./VoiceVisualizer";
import VoiceAnalysisStatus from "./VoiceAnalysisStatus";
import VoiceSignalIndicators from "./VoiceSignalIndicators";

interface VoiceListeningViewProps {
  audioLevel: number;
  isListening: boolean;
  state: "listening" | "processing" | "verified" | "suspicious" | "critical";
  speakerSimilarity?: number | null;
  snrDb?: number | null;
  spoofScore?: number | null;
  hasReport?: boolean;
  onStop: () => void;
  onClose: () => void;
}

export default function VoiceListeningView({
  audioLevel,
  isListening,
  state,
  speakerSimilarity,
  snrDb,
  spoofScore,
  hasReport,
  onStop,
  onClose,
}: VoiceListeningViewProps) {
  return (
    <div className="relative min-h-[90vh] flex flex-col justify-between items-center px-4 sm:px-6 py-6 max-w-xl w-full mx-auto select-none animate-in fade-in duration-300">
      {/* Top Header */}
      <header className="w-full flex items-center justify-between z-20">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close voice analysis"
          className="w-10 h-10 rounded-full border border-white/[0.08] bg-white/[0.03] backdrop-blur-md flex items-center justify-center text-slate-400 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-sm tracking-tight text-white">VaniRakshak</span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-cyan-500/30 bg-cyan-950/30 font-mono text-[10px] text-cyan-300 uppercase tracking-wider font-semibold">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          <span>LIVE ANALYSIS</span>
        </div>
      </header>

      {/* Central Visualizer & Dynamic Status Area */}
      <main className="flex-1 flex flex-col items-center justify-center my-6 sm:my-8 w-full z-10">
        {/* Real-time Dynamic Waveform & Orb */}
        <div className="relative mb-6 sm:mb-8">
          <VoiceVisualizer
            audioLevel={audioLevel}
            isListening={isListening}
            state={state}
          />
        </div>

        {/* Dynamic Status Text */}
        <VoiceAnalysisStatus
          isListening={isListening}
          audioLevel={audioLevel}
          className="mb-8"
        />

        {/* Live Forensic Telemetry Meters */}
        <VoiceSignalIndicators
          audioLevel={audioLevel}
          speakerSimilarity={speakerSimilarity}
          snrDb={snrDb}
          spoofScore={spoofScore}
          hasReport={hasReport}
        />
      </main>

      {/* Bottom Hero Action: Stop & View Analysis */}
      <footer className="w-full flex flex-col items-center z-20 pb-4">
        <button
          type="button"
          onClick={onStop}
          aria-label="Stop voice analysis"
          className="group relative flex items-center gap-3 py-3.5 px-8 rounded-full border border-white/20 bg-gradient-to-b from-white/[0.12] to-white/[0.04] backdrop-blur-xl shadow-[0_0_40px_rgba(255,255,255,0.1)] hover:shadow-[0_0_50px_rgba(6,182,212,0.3)] hover:border-cyan-400/40 hover:scale-102 active:scale-98 transition-all cursor-pointer"
        >
          <div className="w-4 h-4 rounded-sm bg-rose-500 flex items-center justify-center group-hover:scale-110 transition-transform">
            <Square className="w-2.5 h-2.5 text-white fill-white" />
          </div>
          <span className="text-sm font-medium tracking-tight text-white">Stop & View Analysis</span>
        </button>

        <span className="mt-3 text-[11px] font-mono text-slate-400 tracking-wider uppercase">
          16 kHz 16-BIT PCM · DPDP §8(7) COMPLIANT
        </span>
      </footer>
    </div>
  );
}
