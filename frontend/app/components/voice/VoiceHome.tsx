"use client";

import React from "react";
import { Shield, ShieldCheck, History, Sliders } from "lucide-react";
import VoiceMicButton from "./VoiceMicButton";

interface VoiceHomeProps {
  backendOnline: boolean;
  claimedIdentity?: string;
  onStartListening: () => void;
  onOpenHistory: () => void;
  onOpenConsole: () => void;
  className?: string;
}

export default function VoiceHome({
  backendOnline,
  claimedIdentity,
  onStartListening,
  onOpenHistory,
  onOpenConsole,
  className = "",
}: VoiceHomeProps) {
  return (
    <div className={`relative min-h-[90vh] flex flex-col justify-between items-center px-4 sm:px-6 py-6 max-w-xl w-auto mx-auto select-none ${className}`}>
      {/* Top Header */}
      <header className="w-full flex items-center justify-between z-20">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-md flex items-center justify-center">
            <Shield className="w-5 h-5 text-cyan-400" />
          </div>
          <div className="flex flex-col">
            <span className="font-semibold text-base tracking-tight text-white">VaniRakshak</span>
            <span className="font-mono text-[9px] uppercase tracking-wider text-slate-400">
              Voice Authenticity System
            </span>
          </div>
        </div>

        {/* Status indicator badge */}
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-white/[0.08] bg-white/[0.03] backdrop-blur-md font-mono text-[11px] text-slate-300">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              backendOnline ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]" : "bg-rose-400"
            }`}
          />
          <span className="font-medium">{backendOnline ? "SYSTEM READY" : "ENGINE OFFLINE"}</span>
        </div>
      </header>

      {/* Main Center Stage: Hero Microphone Interaction */}
      <main className="flex-1 flex flex-col items-center justify-center my-8 w-full z-10 text-center">
        {/* Enrolled Target Anchor Pill (if present) */}
        {claimedIdentity && (
          <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full border border-cyan-500/20 bg-cyan-950/20 backdrop-blur-md font-mono text-[11px] text-cyan-300 mb-8">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>Target Anchor: {claimedIdentity.replace(/_/g, " ")}</span>
          </div>
        )}

        {/* Hero Mic Button */}
        <div className="my-2">
          <VoiceMicButton
            onClick={onStartListening}
            isReady={backendOnline}
            size="large"
          />
        </div>

        {/* Hero Headline and Concise Supporting Text */}
        <div className="mt-8 max-w-xs space-y-2">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-white">
            Tap to verify
          </h1>
          <p className="text-slate-300 text-xs sm:text-sm font-normal leading-relaxed">
            Analyze a voice for authenticity and deepfake risk.
          </p>
        </div>
      </main>

      {/* Bottom Bar with Secondary Controls */}
      <footer className="w-full flex items-center justify-between z-20 pb-4 pt-2">
        <button
          type="button"
          onClick={onOpenHistory}
          className="flex items-center gap-1.5 px-4 py-2 rounded-full border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] text-xs text-slate-400 hover:text-white transition-colors cursor-pointer"
        >
          <History className="w-3.5 h-3.5" />
          <span>Recent Sessions</span>
        </button>

        <button
          type="button"
          onClick={onOpenConsole}
          className="flex items-center gap-1.5 px-4 py-2 rounded-full border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] text-xs text-slate-400 hover:text-white transition-colors cursor-pointer"
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>Forensic Console</span>
        </button>
      </footer>
    </div>
  );
}
