"use client";

import React from "react";
import { AlertOctagon, Lock, RotateCcw, ShieldAlert, PhoneCall } from "lucide-react";

interface CriticalResultProps {
  riskScore: number;
  verdict?: string;
  onReset: () => void;
  onViewDeepDetails?: () => void;
}

export default function CriticalResult({
  riskScore,
  verdict = "SYNTHETIC_VOICE_CLONE",
  onReset,
  onViewDeepDetails,
}: CriticalResultProps) {
  return (
    <div className="flex flex-col items-center text-center max-w-md w-full mx-auto px-4 py-6 animate-in fade-in zoom-in-95 duration-400">
      {/* Top Security Alert Badge */}
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-rose-500/40 bg-rose-500/15 font-mono text-[11px] font-semibold text-rose-400 tracking-wider uppercase mb-6">
        <ShieldAlert className="w-3.5 h-3.5" />
        <span>AUTONOMOUS INTERLOCK ENGAGED</span>
      </div>

      {/* Central Warning Graphic with Rose Glow */}
      <div className="relative mb-6">
        <div className="absolute -inset-4 rounded-full bg-rose-500/25 blur-2xl animate-pulse" />
        <div className="relative w-28 h-28 sm:w-32 sm:h-32 rounded-full border-2 border-rose-500/50 bg-gradient-to-b from-rose-950/50 to-[#1e070b] flex items-center justify-center shadow-[0_0_50px_rgba(244,63,94,0.35)]">
          <div className="w-16 h-16 rounded-full bg-rose-500/20 border border-rose-400/40 flex items-center justify-center">
            <AlertOctagon className="w-9 h-9 text-rose-300" strokeWidth={2.4} />
          </div>
        </div>
      </div>

      {/* Primary Headline & Copy */}
      <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-white mb-2">
        VOICE RISK DETECTED
      </h2>
      <p className="text-rose-200/90 text-xs sm:text-sm mb-6 max-w-xs leading-relaxed font-normal">
        VaniRakshak detected a high-risk synthetic voice or biometric spoofing signal ({riskScore}% Risk).
      </p>

      {/* Autonomous Interlock Notice Card */}
      <div className="w-full rounded-2xl border border-rose-500/30 bg-rose-950/20 backdrop-blur-lg p-5 mb-6 text-left shadow-xl space-y-3.5">
        <div className="flex items-center justify-between pb-3 border-b border-rose-500/20">
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4 text-rose-400" />
            <span className="text-xs font-semibold text-white tracking-tight">GATEWAY CIRCUIT LOCKED</span>
          </div>
          <span className="px-2 py-0.5 rounded-full border border-rose-500/30 bg-rose-500/20 font-mono text-[10px] text-rose-300 font-semibold uppercase">
            BLOCKED
          </span>
        </div>

        <p className="text-xs text-rose-200/80 leading-relaxed">
          Transaction and authorization controls have been autonomously suspended. Do not authorize fund transfers, OTP disclosures, or remote screen sharing.
        </p>

        <div className="pt-2 border-t border-rose-500/20 flex items-center justify-between text-[11px] font-mono">
          <span className="text-slate-400">DETECTED SIGNATURE:</span>
          <span className="text-rose-300 font-medium">{verdict.replace(/_/g, " ")}</span>
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
            <span>Audit Trail</span>
          </button>
        )}
      </div>

      {/* Emergency Contact Reminder */}
      <div className="mt-4 flex items-center gap-1.5 text-xs text-slate-400">
        <PhoneCall className="w-3.5 h-3.5 text-rose-400" />
        <span>Suspected extortion or fraud? Dial 1930 Cyber Helpline</span>
      </div>
    </div>
  );
}
