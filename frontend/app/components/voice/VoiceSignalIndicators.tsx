"use client";

import React from "react";

interface VoiceSignalIndicatorsProps {
  audioLevel: number; // 0 to 100
  speakerSimilarity?: number | null; // e.g. ASV consistency 0.0 to 1.0
  snrDb?: number | null; // e.g. 18.4 dB
  spoofScore?: number | null; // e.g. p_spoof 0.0 to 1.0
  hasReport?: boolean;
  className?: string;
}

export default function VoiceSignalIndicators({
  audioLevel,
  speakerSimilarity = null,
  snrDb = null,
  spoofScore = null,
  hasReport = false,
  className = "",
}: VoiceSignalIndicatorsProps) {
  // 1. Live Voice Signal based on microphone PCM RMS
  const voiceSignalPercent = Math.min(100, Math.max(10, Math.round(audioLevel * 1.2)));

  // 2. Speaker Match
  const hasSpeakerMatch = speakerSimilarity !== null && speakerSimilarity !== undefined;
  const speakerMatchPercent = hasSpeakerMatch ? Math.round(speakerSimilarity * 100) : null;

  // 3. Audio Quality based on SNR (0-30 dB normalized)
  const hasSnr = snrDb !== null && snrDb !== undefined;
  const audioQualityPercent = hasSnr ? Math.min(100, Math.max(20, Math.round((snrDb / 30) * 100))) : null;

  // 4. Spoof Signal based on p_spoof
  const hasSpoof = spoofScore !== null && spoofScore !== undefined;
  const spoofPercent = hasSpoof ? Math.round(spoofScore * 100) : null;

  return (
    <div className={`grid grid-cols-2 sm:grid-cols-4 gap-2.5 max-w-lg w-full ${className}`}>
      {/* 1. Voice Signal */}
      <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-md p-2.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[10px] font-mono tracking-wider text-slate-400">
          <span>VOICE SIGNAL</span>
          <span className="text-cyan-400 font-semibold">{voiceSignalPercent}%</span>
        </div>
        <div className="mt-2 h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-150 rounded-full"
            style={{ width: `${voiceSignalPercent}%` }}
          />
        </div>
      </div>

      {/* 2. Speaker Match */}
      <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-md p-2.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[10px] font-mono tracking-wider text-slate-400">
          <span>SPEAKER MATCH</span>
          <span className={hasSpeakerMatch ? "text-emerald-400 font-semibold" : "text-slate-500"}>
            {hasSpeakerMatch ? `${speakerMatchPercent}%` : "Analyzing…"}
          </span>
        </div>
        <div className="mt-2 h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              hasSpeakerMatch
                ? (speakerMatchPercent ?? 0) >= 60
                  ? "bg-emerald-400"
                  : "bg-amber-400"
                : "bg-slate-700 animate-pulse w-1/3"
            }`}
            style={{ width: hasSpeakerMatch ? `${speakerMatchPercent}%` : undefined }}
          />
        </div>
      </div>

      {/* 3. Audio Quality */}
      <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-md p-2.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[10px] font-mono tracking-wider text-slate-400">
          <span>AUDIO QUALITY</span>
          <span className={hasSnr ? "text-blue-400 font-semibold" : "text-slate-500"}>
            {hasSnr ? `${snrDb?.toFixed(1)} dB` : hasReport ? "Normal" : "Analyzing…"}
          </span>
        </div>
        <div className="mt-2 h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              hasSnr ? "bg-blue-400" : "bg-slate-700 animate-pulse w-1/2"
            }`}
            style={{ width: hasSnr ? `${audioQualityPercent}%` : undefined }}
          />
        </div>
      </div>

      {/* 4. Spoof Signal */}
      <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-md p-2.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-[10px] font-mono tracking-wider text-slate-400">
          <span>SPOOF SIGNAL</span>
          <span
            className={
              hasSpoof
                ? (spoofPercent ?? 0) >= 70
                  ? "text-rose-400 font-semibold"
                  : (spoofPercent ?? 0) >= 35
                  ? "text-amber-400 font-semibold"
                  : "text-emerald-400 font-semibold"
                : "text-slate-500"
            }
          >
            {hasSpoof ? `${spoofPercent}%` : "Analyzing…"}
          </span>
        </div>
        <div className="mt-2 h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              hasSpoof
                ? (spoofPercent ?? 0) >= 70
                  ? "bg-rose-500"
                  : (spoofPercent ?? 0) >= 35
                  ? "bg-amber-400"
                  : "bg-emerald-400"
                : "bg-slate-700 animate-pulse w-1/4"
            }`}
            style={{ width: hasSpoof ? `${spoofPercent}%` : undefined }}
          />
        </div>
      </div>
    </div>
  );
}
