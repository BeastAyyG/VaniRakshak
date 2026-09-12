"use client";

import React from "react";
import { X, ShieldCheck, AlertTriangle, AlertOctagon, Clock } from "lucide-react";

export interface VoiceHistoryEntry {
  id: string;
  timestamp: string;
  verdict: string;
  tier: "ALLOW" | "CHALLENGE" | "BLOCK";
  riskScore: number;
  durationSec?: number;
}

interface VoiceHistorySheetProps {
  isOpen: boolean;
  history: VoiceHistoryEntry[];
  onClose: () => void;
  onClear: () => void;
}

export default function VoiceHistorySheet({
  isOpen,
  history,
  onClose,
  onClear,
}: VoiceHistorySheetProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-md rounded-3xl border border-white/[0.1] bg-[#091522] p-6 shadow-2xl space-y-5 animate-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-cyan-400" />
            <h3 className="font-semibold text-base text-white">Recent Voice Sessions</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 rounded-full border border-white/[0.08] flex items-center justify-center text-slate-400 hover:text-white transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* History List */}
        <div className="max-h-[320px] overflow-y-auto space-y-2.5 pr-1">
          {history.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-xs font-mono">
              No recent voice analysis sessions recorded.
            </div>
          ) : (
            history.map((item) => (
              <div
                key={item.id}
                className="p-3.5 rounded-2xl border border-white/[0.06] bg-white/[0.02] flex items-center justify-between"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      item.tier === "BLOCK"
                        ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                        : item.tier === "CHALLENGE"
                        ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                        : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                    }`}
                  >
                    {item.tier === "BLOCK" ? (
                      <AlertOctagon className="w-4 h-4" />
                    ) : item.tier === "CHALLENGE" ? (
                      <AlertTriangle className="w-4 h-4" />
                    ) : (
                      <ShieldCheck className="w-4 h-4" />
                    )}
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-white">
                      {item.tier === "BLOCK"
                        ? "BLOCKED"
                        : item.tier === "CHALLENGE"
                        ? "CHALLENGE REQUIRED"
                        : "VERIFIED AUTHENTIC"}
                    </div>
                    <div className="text-[10px] font-mono text-slate-400">{item.timestamp}</div>
                  </div>
                </div>

                <div className="text-right">
                  <span
                    className={`font-mono text-xs font-semibold ${
                      item.tier === "BLOCK"
                        ? "text-rose-400"
                        : item.tier === "CHALLENGE"
                        ? "text-amber-400"
                        : "text-emerald-400"
                    }`}
                  >
                    {item.riskScore}% Risk
                  </span>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between pt-2 border-t border-white/[0.08] text-xs">
          <button
            type="button"
            onClick={onClear}
            disabled={history.length === 0}
            className="text-slate-500 hover:text-rose-400 disabled:opacity-30 transition-colors cursor-pointer"
          >
            Clear session history
          </button>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-full bg-white text-slate-950 font-medium text-xs hover:bg-slate-200 transition-colors cursor-pointer"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
