"use client";

import React, { useState } from "react";
import { AlertTriangle, Lock, ArrowRight, RotateCcw } from "lucide-react";

interface SuspiciousResultProps {
  riskScore: number;
  challengePhrase?: string;
  expectedDigits?: string[];
  challengeStatus: "none" | "awaiting" | "passed" | "failed";
  onSubmitChallenge: (responseText: string) => Promise<void>;
  onReset: () => void;
}

export default function SuspiciousResult({
  riskScore,
  challengePhrase = "Mango, 8, nadi, 4.",
  challengeStatus,
  onSubmitChallenge,
  onReset,
}: SuspiciousResultProps) {
  const [digitsInput, setDigitsInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!digitsInput.trim() || isSubmitting) return;
    setIsSubmitting(true);
    try {
      await onSubmitChallenge(digitsInput.trim());
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col items-center text-center max-w-md w-full mx-auto px-4 py-6 animate-in fade-in zoom-in-95 duration-400">
      {/* Top Warning Badge */}
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-amber-500/30 bg-amber-500/10 font-mono text-[11px] font-semibold text-amber-400 tracking-wider uppercase mb-6">
        <AlertTriangle className="w-3.5 h-3.5" />
        <span>CHALLENGE INTERLOCK ACTIVE</span>
      </div>

      {/* Central Warning Graphic with Amber Glow */}
      <div className="relative mb-6">
        <div className="absolute -inset-4 rounded-full bg-amber-500/20 blur-xl animate-pulse" />
        <div className="relative w-28 h-28 sm:w-32 sm:h-32 rounded-full border-2 border-amber-400/40 bg-gradient-to-b from-amber-950/40 to-[#1a1205] flex items-center justify-center shadow-[0_0_40px_rgba(245,158,11,0.25)]">
          <div className="w-16 h-16 rounded-full bg-amber-500/20 border border-amber-400/30 flex items-center justify-center">
            <span className="text-3xl font-bold text-amber-300 font-mono">!</span>
          </div>
        </div>
      </div>

      {/* Primary Headline & Copy */}
      <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-white mb-2">
        NEEDS VERIFICATION
      </h2>
      <p className="text-slate-300 text-xs sm:text-sm mb-6 max-w-xs leading-relaxed font-normal">
        The acoustic signal deviates from certified baselines (Risk {riskScore}%). Complete the interactive liveness challenge below.
      </p>

      {/* Active Challenge Interactive Card */}
      <div className="w-full rounded-2xl border border-amber-500/30 bg-amber-950/15 backdrop-blur-lg p-5 mb-6 text-left shadow-xl">
        <div className="flex items-center justify-between pb-3 border-b border-amber-500/20">
          <span className="text-[11px] font-mono text-amber-400/80 font-medium tracking-wider">ANTI-REPLAY LIVENESS DRILL</span>
          <Lock className="w-4 h-4 text-amber-400" />
        </div>

        <div className="my-3">
          <div className="text-[11px] font-mono text-slate-400 mb-1">PROMPT PHRASE:</div>
          <div className="p-3 rounded-xl border border-amber-400/20 bg-black/40 font-mono text-sm sm:text-base text-amber-200 tracking-wide font-medium">
            &ldquo;{challengePhrase}&rdquo;
          </div>
        </div>

        <p className="text-xs text-slate-400 mb-3 leading-relaxed">
          Instruct the speaker to repeat <span className="text-white font-semibold underline">ONLY the numbers/digits</span> in the exact sequence immediately.
        </p>

        {/* Input response field */}
        {challengeStatus === "passed" ? (
          <div className="p-3 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-medium text-center">
            ✓ Challenge Passed — Human latency & digit sequence confirmed.
          </div>
        ) : challengeStatus === "failed" ? (
          <div className="p-3 rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-medium text-center">
            ✕ Challenge Failed — Digit mismatch or synthetic latency. Escalated to BLOCK.
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={digitsInput}
              onChange={(e) => setDigitsInput(e.target.value)}
              placeholder="e.g. 8 4"
              className="flex-1 bg-black/50 border border-white/10 rounded-xl px-3.5 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-amber-400/50"
            />
            <button
              type="submit"
              disabled={!digitsInput.trim() || isSubmitting}
              className="px-4 py-2 bg-amber-500 hover:bg-amber-400 disabled:opacity-40 text-slate-950 font-semibold text-xs rounded-xl flex items-center gap-1 transition-all cursor-pointer"
            >
              <span>Submit</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </form>
        )}
      </div>

      {/* Action Buttons */}
      <button
        type="button"
        onClick={onReset}
        className="flex items-center justify-center gap-2 py-3 px-6 rounded-full border border-white/[0.12] bg-white/[0.04] text-slate-300 hover:text-white hover:bg-white/[0.08] text-xs font-medium transition-all cursor-pointer"
      >
        <RotateCcw className="w-3.5 h-3.5" />
        <span>Analyze another voice</span>
      </button>
    </div>
  );
}
