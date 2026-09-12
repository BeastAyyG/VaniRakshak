"use client";

import React from "react";
import { MicOff, ShieldAlert, ArrowRight } from "lucide-react";

interface MicrophonePermissionProps {
  onRetry: () => void;
  onCancel: () => void;
  errorMessage?: string;
}

export default function MicrophonePermission({
  onRetry,
  onCancel,
  errorMessage = "Microphone access is required to analyze voice authenticity.",
}: MicrophonePermissionProps) {
  return (
    <div className="flex flex-col items-center text-center max-w-sm w-full mx-auto px-6 py-8 animate-in fade-in duration-300">
      {/* Mic Icon with Warning Halo */}
      <div className="relative mb-6">
        <div className="absolute -inset-3 rounded-full bg-rose-500/20 blur-lg" />
        <div className="relative w-20 h-20 rounded-full border border-rose-500/30 bg-rose-950/30 flex items-center justify-center">
          <MicOff className="w-9 h-9 text-rose-400" />
        </div>
      </div>

      <h3 className="text-xl font-semibold text-white mb-2">Microphone Access Blocked</h3>
      <p className="text-xs sm:text-sm text-slate-300 mb-6 leading-relaxed">
        {errorMessage} Please grant microphone permission in your browser or device settings to start live forensic streaming.
      </p>

      <div className="w-full space-y-2.5">
        <button
          type="button"
          onClick={onRetry}
          className="w-full py-3 px-5 rounded-full bg-white text-slate-950 font-medium text-xs sm:text-sm hover:bg-slate-200 transition-all cursor-pointer flex items-center justify-center gap-1.5 shadow-lg"
        >
          <span>Grant Microphone Access</span>
          <ArrowRight className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={onCancel}
          className="w-full py-2.5 px-4 text-xs text-slate-400 hover:text-white transition-colors cursor-pointer"
        >
          Return to Home
        </button>
      </div>

      <div className="mt-6 flex items-center gap-1.5 text-[11px] font-mono text-slate-400">
        <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
        <span>DPDP Act 2023: Zero Audio Retention</span>
      </div>
    </div>
  );
}
