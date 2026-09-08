"use client";

import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, UserX, Play, CheckCircle2 } from "lucide-react";

export type Verdict =
  | "AUTHENTIC_TARGET"
  | "SYNTHETIC_VOICE_CLONE"
  | "SUSPICIOUS_SPLICED_AUDIO"
  | "IMPOSTOR_SPEAKER";

export interface BenchmarkItem {
  id: string;
  title: string;
  subtitle: string;
  fileName: string;
  audioUrl: string;
  /** Ground-truth label from data provenance. `null` = no honest ground truth. */
  expectedVerdict: Verdict | null;
  /** What the shipped pipeline actually returned when these were measured. */
  measuredVerdict: Verdict;
  measuredRisk: number;
  measuredSpeakerSim: number;
  badge: string;
  color: "emerald" | "rose" | "amber" | "purple";
  description: string;
  keyDetails: string[];
  /** Shown verbatim in the UI; anything a reader could otherwise over-read. */
  caveat?: string;
}

/**
 * Every number below was measured by `python scripts/measure_demos.py` against
 * the shipped pipeline, not estimated. Clips are rebuilt by
 * `scripts/build_demo_suite.py` from audio held out of anchor enrolment.
 */
export const BENCHMARKS: BenchmarkItem[] = [
  {
    id: "demo_01",
    title: "Authentic target voice.",
    subtitle: "Reference target speaker",
    fileName: "demo_01_genuine.wav",
    audioUrl: "/demo/demo_01_genuine.wav",
    expectedVerdict: "AUTHENTIC_TARGET",
    measuredVerdict: "AUTHENTIC_TARGET",
    measuredRisk: 33.9,
    measuredSpeakerSim: 0.7282,
    badge: "BENCHMARK 1 • BONA FIDE",
    color: "emerald",
    description:
      "Real speech from the enrolled target speaker, held out of anchor enrolment so the embedding has never seen these clips.",
    keyDetails: [
      "Speaker similarity 0.73 vs anchor",
      "Risk 33.9/100 — ALLOW band",
      "Held out of enrolment (3 clips)",
    ],
  },
  {
    id: "demo_02",
    title: "Degraded copy of target.",
    subtitle: "Signal distortion, not a clone",
    fileName: "demo_02_degraded_copy.wav",
    audioUrl: "/demo/demo_02_degraded_copy.wav",
    expectedVerdict: "AUTHENTIC_TARGET",
    measuredVerdict: "AUTHENTIC_TARGET",
    measuredRisk: 34.1,
    measuredSpeakerSim: 0.7026,
    badge: "BENCHMARK 2 • DEGRADED",
    color: "rose",
    description:
      "The same target speech put through tanh saturation and a 6% resample. The voice is still the target's, so 'authentic' is the correct answer — this is a distortion, not an identity attack.",
    keyDetails: [
      "tanh saturation + 6% resample",
      "Speaker similarity 0.70 vs anchor",
      "Risk 34.1/100 — ALLOW band",
    ],
    caveat:
      "Not a neural voice clone. The shipped countermeasure (CM) detector is a stub that has never seen synthetic audio, so this file does NOT test clone detection.",
  },
  {
    id: "demo_03",
    title: "Partial splice injection.",
    subtitle: "Target / non-target composite",
    fileName: "demo_03_partial_splice.wav",
    audioUrl: "/demo/demo_03_partial_splice.wav",
    expectedVerdict: "SUSPICIOUS_SPLICED_AUDIO",
    measuredVerdict: "SUSPICIOUS_SPLICED_AUDIO",
    measuredRisk: 51.4,
    measuredSpeakerSim: 0.5584,
    badge: "BENCHMARK 3 • SPLICED",
    color: "amber",
    description:
      "Target speech with a different speaker spliced in at 4.0s; the target voice returns at 8.0s. Produces a mid-band risk and a depressed similarity score.",
    keyDetails: [
      "Splice boundaries at 4.0s and 8.0s",
      "Speaker similarity 0.56 — partial mismatch",
      "Risk 51.4/100 — CHALLENGE band",
    ],
  },
  {
    id: "demo_04",
    title: "Zero-shot impostor.",
    subtitle: "Unenrolled identity fraud",
    fileName: "demo_04_impostor.wav",
    audioUrl: "/demo/demo_04_impostor.wav",
    expectedVerdict: "IMPOSTOR_SPEAKER",
    measuredVerdict: "IMPOSTOR_SPEAKER",
    measuredRisk: 67.0,
    measuredSpeakerSim: 0.0903,
    badge: "BENCHMARK 4 • IMPOSTOR",
    color: "purple",
    description:
      "A genuinely different, unenrolled speaker. Similarity collapses to 0.09 against the anchor and risk rises into the CHALLENGE band.",
    keyDetails: [
      "Speaker similarity 0.09 vs anchor",
      "Risk 67.0/100 — CHALLENGE band",
      "Zero-retention audit logged",
    ],
    caveat:
      "Similarity comes from the numpy Welch-PSD speaker stub with a fitted logistic calibration, not ECAPA-TDNN. The trained model is not loaded in this environment.",
  },
];

interface DemoQuickSelectorProps {
  selectedId: string | null;
  onSelect: (item: BenchmarkItem) => void;
  isAnalyzing: boolean;
}

export default function DemoQuickSelector({ selectedId, onSelect, isAnalyzing }: DemoQuickSelectorProps) {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 border-b border-white/[0.06] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-wider text-[#a1a1a1]">
              STANDARDIZED BENCHMARK SUITE
            </span>
          </div>
          <h3 className="mt-1 text-lg font-semibold tracking-[-0.6px] text-white">
            Pre-configured evaluation files.
          </h3>
          <p className="mt-0.5 text-xs text-[#888888]">
            One-click evaluation files covering bona fide, signal-degraded copy, splice tampering, and impostor vectors. Figures shown are measured, not estimated.
          </p>
        </div>
        <span className="self-start sm:self-auto rounded-full border border-white/[0.08] bg-[#171717] px-3 py-1 font-mono text-[11px] text-[#a1a1a1]">
          16 kHz mono • Enrolled: Narendra Modi
        </span>
      </div>

      <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-4">
        {BENCHMARKS.map((item) => {
          const isSelected = selectedId === item.id;
          const colorStyles = {
            emerald: {
              border: isSelected ? "border-emerald-500/80 bg-[#111814]" : "border-white/[0.08] hover:border-emerald-500/40 bg-[#121212]",
              badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
              icon: <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />,
              btn: isSelected ? "bg-emerald-500 text-black font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
            rose: {
              border: isSelected ? "border-rose-500/80 bg-[#1a1113]" : "border-white/[0.08] hover:border-rose-500/40 bg-[#121212]",
              badge: "border-rose-500/30 bg-rose-500/10 text-rose-400",
              icon: <ShieldAlert className="h-4 w-4 text-rose-400 shrink-0" />,
              btn: isSelected ? "bg-rose-500 text-white font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
            amber: {
              border: isSelected ? "border-amber-500/80 bg-[#1a1610]" : "border-white/[0.08] hover:border-amber-500/40 bg-[#121212]",
              badge: "border-amber-500/30 bg-amber-500/10 text-amber-400",
              icon: <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />,
              btn: isSelected ? "bg-amber-500 text-black font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
            purple: {
              border: isSelected ? "border-purple-500/80 bg-[#17111a]" : "border-white/[0.08] hover:border-purple-500/40 bg-[#121212]",
              badge: "border-purple-500/30 bg-purple-500/10 text-purple-400",
              icon: <UserX className="h-4 w-4 text-purple-400 shrink-0" />,
              btn: isSelected ? "bg-purple-500 text-white font-semibold" : "bg-white text-black hover:bg-neutral-200",
            },
          }[item.color];

          return (
            <div
              key={item.id}
              onClick={() => !isAnalyzing && onSelect(item)}
              className={`relative flex flex-col justify-between rounded-lg border p-4 transition-all duration-150 cursor-pointer ${colorStyles.border} ${
                isSelected ? "ring-1 ring-white/20" : ""
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2.5">
                  <span className={`rounded-full border px-2 py-0.5 font-mono text-[9.5px] font-medium tracking-wider uppercase ${colorStyles.badge}`}>
                    {item.badge}
                  </span>
                  {isSelected && (
                    <span className="flex items-center gap-1 font-mono text-[10px] text-white">
                      <CheckCircle2 className="h-3 w-3 text-emerald-400" /> Active
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 mb-2">
                  <div className="p-1 rounded bg-black/40 border border-white/[0.06]">
                    {colorStyles.icon}
                  </div>
                  <div>
                    <h4 className="font-semibold text-sm tracking-[-0.3px] text-white">
                      {item.title}
                    </h4>
                    <p className="font-mono text-[11px] text-[#888888]">
                      {item.subtitle}
                    </p>
                  </div>
                </div>

                <p className="text-xs leading-relaxed text-[#a1a1a1] mt-2.5">
                  {item.description}
                </p>

                {item.caveat && (
                  <p className="mt-2 rounded-md border border-amber-500/20 bg-amber-500/[0.06] px-2 py-1.5 text-[10.5px] leading-relaxed text-amber-300/90">
                    {item.caveat}
                  </p>
                )}

                <div className="mt-3 space-y-1 border-t border-white/[0.06] pt-2.5 font-mono text-[10.5px] text-[#888888]">
                  {item.keyDetails.map((detail, idx) => (
                    <div key={idx} className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-[#555]" />
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 pt-1">
                <button
                  type="button"
                  disabled={isAnalyzing}
                  className={`w-full flex items-center justify-center gap-1.5 rounded-full py-1.5 px-3 text-xs font-medium transition shadow-sm ${colorStyles.btn} disabled:opacity-40 cursor-pointer`}
                >
                  <Play className="h-3 w-3 fill-current" />
                  <span>{isSelected ? "Re-evaluate" : "Run evaluation"}</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
