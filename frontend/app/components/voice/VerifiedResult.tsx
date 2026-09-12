"use client";

import React from "react";
import { Check, ShieldCheck, Activity, RotateCcw } from "lucide-react";

interface VerifiedResultProps {
  riskScore: number;
  speakerSimilarity?: number | null;
  snrDb?: number | null;
  onReset: () => void;
  onViewDeepDetails?: () => void;
}

export default function VerifiedResult({
  riskScore,
  speakerSimilarity,
  snrDb,
  onReset,
  onViewDeepDetails,
}: VerifiedResultProps) {
  const speakerMatchLabel =
    speakerSimilarity !== null && speakerSimilarity !== undefined
      ? speakerSimilarity >= 0.75
        ? "High Match"
        : `${Math.round(speakerSimilarity * 100)}% Match`
      : "Baseline Verified";

  return (
    <div className="flex flex-col items-center text-center max-w-md w-full mx-auto px-4 py-6 animate-in fade-in zoom-in-95 duration-400">
      {/* Top Category Badge */}
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 font-mono text-[11px] font-semibold text-emerald-400 tracking-wider uppercase mb-6">
        <ShieldCheck className="w-3.5 h-3.5" />
        <span>AUTHENTIC VOICE SIGNAL</span>
      </div>

      {/* Central Verified Graphic with Aura */}
      <div className="relative mb-6">
        <div className="absolute -inset-4 rounded-full bg-emerald-500/20 blur-xl animate-pulse" />
        <div className="relative w-28 h-28 sm:w-32 sm:h-32 rounded-full border-2 border-emerald-400/40 bg-gradient-to-b from-emerald-950/40 to-[#041612] flex items-center justify-center shadow-[0_0_40px_rgba(16,185,129,0.25)]">
          <div className="w-16 h-16 rounded-full bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center">
            <Check className="w-9 h-9 text-emerald-300" strokeWidth={2.8} />
          </div>
        </div>
      </div>

      {/* Primary Headline & Copy */}
      <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white mb-2">
        VERIFIED
      </h2>
      <p className="text-slate-300 text-sm sm:text-base mb-8 max-w-xs leading-relaxed font-normal">
        This voice appears authentic and aligns with trusted biometric criteria.
      </p>

      {/* Supporting Forensic Intelligence Summary Card */}
      <div className="w-full rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-lg p-5 mb-8 text-left space-y-4 shadow-xl">
        <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
          <span className="text-xs text-slate-400 font-mono">FORENSIC RISK LEVEL</span>
          <span className="text-sm font-semibold text-emerald-400 flex items-center gap-1">
            <span>LOW RISK</span>
            <span className="text-xs font-mono font-normal text-slate-400">({riskScore}%)</span>
          </span>
        </div>

        <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
          <span className="text-xs text-slate-400 font-mono">SPEAKER CONSISTENCY</span>
          <span className="text-sm font-medium text-slate-200">{speakerMatchLabel}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-slate-400 font-mono">AUDIO FIDELITY</span>
          <span className="text-sm font-medium text-slate-200">
            {snrDb ? `${snrDb.toFixed(1)} dB SNR` : "Certified Studio Quality"}
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 w-full">
        <button
          type="button"
          onClick={onReset}
          className="flex-1 flex items-center justify-center gap-2 py-3.5 px-6 rounded-full bg-white text-slate-950 font-medium text-sm tracking-tight hover:bg-slate-200 active:scale-98 transition-all cursor-pointer shadow-lg"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Analyze another voice</span>
        </button>

        {onViewDeepDetails && (
          <button
            type="button"
            onClick={onViewDeepDetails}
            className="flex items-center justify-center gap-2 py-3.5 px-5 rounded-full border border-white/[0.12] bg-white/[0.04] text-slate-300 hover:text-white hover:bg-white/[0.08] text-sm font-medium transition-all cursor-pointer"
          >
            <Activity className="w-4 h-4" />
            <span>Forensic Console</span>
          </button>
        )}
      </div>
    </div>
  );
}
