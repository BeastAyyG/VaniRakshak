"use client";

import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, UserX, Download, Lock, Unlock, Cpu, MessageSquareWarning } from "lucide-react";
import { WindowResult } from "./WaveformEvidenceTimeline";

export interface ForensicReportData {
  session_id: string;
  audio_sha256: string;
  duration_sec: number;
  device_used: string;
  windows_count: number;
  overall_risk_score: number;
  verdict: "AUTHENTIC_TARGET" | "SYNTHETIC_VOICE_CLONE" | "SUSPICIOUS_SPLICED_AUDIO" | "IMPOSTOR_SPEAKER" | string;
  windows?: WindowResult[];
  mean_acoustic_anomaly?: number;
  max_acoustic_anomaly?: number;
  /**
   * Whether ASV actually ran a real biometric comparison this report (the
   * backend's `asv_evaluated` / `metrics.asv_evaluated`). When false,
   * `mean_speaker_sim` is the fusion-neutral 0.5 placeholder, not a
   * measured similarity — see vanirakshak/detectors/pipeline.py. Optional
   * so a report built before this field existed still renders (as
   * "evaluated", matching the old always-render-a-number behavior).
   */
  asv_evaluated?: boolean;
  mean_speaker_sim?: number | null;
  mean_micro_tremor?: number | null;
  mean_spectral_flatness?: number | null;
  mean_high_band_flatness?: number | null;
  mean_zcr_variance?: number | null;
  vocoder_phase_discontinuities_detected?: number;
  detected_boundary_timestamps?: number[];
  dpdp_compliance?: {
    receipt_sha256: string;
    statutory_act?: string;
    zero_retention_verified: boolean;
    timestamp: string;
  };
  /**
   * Real (non-stub) signal from the trained ml/ model: an IDENTITY check
   * ("is this the enrolled voice?"). Deliberately kept separate from
   * `verdict` — it is NOT a clone/synthetic decision.
   */
  target_speaker?: {
    available: boolean;
    label: string;
    is_target: boolean;
    prob_target?: number | null;
    llr?: number | null;
    reason?: string | null;
  } | null;

  /**
   * Parallel ASR + scam-keyword context signal (vanirakshak/semantic/).
   * Advisory only — a small additive term in the acoustic risk fusion, never
   * the source of `verdict` itself. See docs/PROTOCOL.md §3.1.
   */
  semantic?: {
    available: boolean;
    transcript?: string;
    hinglish_text?: string;
    matches?: { category: string; term: string }[];
    categories_hit?: string[];
    semantic_risk?: number;
    /** Bag-of-words scam vocabulary score, before claim verification. */
    keyword_risk?: number;
    /** Max severity among fired policy-contradiction rules (claims.py). */
    claim_risk?: number;
    /** Specific verified contradictions, each citable to an authority. */
    claim_findings?: {
      rule_id: string;
      claim: string;
      finding: string;
      authority: string;
      severity: number;
      matched_terms?: string[];
    }[];
    reason?: string | null;
  } | null;
}

interface ForensicVerdictCardProps {
  report: ForensicReportData | null;
  onDownloadReport: () => void;
}

export default function ForensicVerdictCard({ report, onDownloadReport }: ForensicVerdictCardProps) {
  if (!report) return null;

  const risk = report.overall_risk_score ?? 0;

  const verdictConfig = {
    AUTHENTIC_TARGET: {
      title: "Authentic target speaker verified.",
      subtitle: "Bona fide voice biometrics",
      badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
      dialStroke: "#10b981",
      icon: <ShieldCheck className="h-5 w-5 text-emerald-400" />,
      description: "Audio displays natural vocal tract micro-tremor, consistent biomechanics, and robust cosine similarity to the enrolled target anchor.",
      interlockStatus: "TRANSACTION AUTHORIZED",
      interlockLocked: false,
    },
    SYNTHETIC_VOICE_CLONE: {
      title: "Synthetic voice clone detected.",
      subtitle: "Neural vocoder impersonation",
      badge: "border-rose-500/30 bg-rose-500/10 text-rose-400",
      dialStroke: "#f43f5e",
      icon: <ShieldAlert className="h-5 w-5 text-rose-400" />,
      description: "Unnatural high-frequency spectral flattening (>4 kHz) and latent vector collapse detected, characteristic of neural vocoder TTS models.",
      interlockStatus: "TRANSFER FROZEN",
      interlockLocked: true,
    },
    SUSPICIOUS_SPLICED_AUDIO: {
      title: "Suspicious audio splice detected.",
      subtitle: "Localized phase discontinuity",
      badge: "border-amber-500/30 bg-amber-500/10 text-amber-400",
      dialStroke: "#f59e0b",
      icon: <AlertTriangle className="h-5 w-5 text-amber-400" />,
      description: "Phase boundary discontinuities or segment variance anomalies detected mid-recording, indicating genuine audio spliced with synthetic insertions.",
      interlockStatus: "SECURITY TRIPWIRE ACTIVE",
      interlockLocked: true,
    },
    IMPOSTOR_SPEAKER: {
      title: "Biometric impostor speaker rejected.",
      subtitle: "Unenrolled voice identity fraud",
      badge: "border-purple-500/30 bg-purple-500/10 text-purple-400",
      dialStroke: "#a855f7",
      icon: <UserX className="h-5 w-5 text-purple-400" />,
      description: "Speaker vocal tract acoustic embedding does not match the enrolled anchor profile (ECAPA-TDNN similarity below the 0.55 threshold).",
      interlockStatus: "ACCESS REJECTED",
      interlockLocked: true,
    },
  }[report.verdict] || {
    title: `${report.verdict}.`,
    subtitle: "Evaluation complete",
    badge: "border-white/[0.08] bg-[#171717] text-white",
    dialStroke: "#888888",
    icon: <Cpu className="h-5 w-5 text-[#888888]" />,
    description: "Multi-factor telemetry evaluated across 2.0s sliding windows.",
    interlockStatus: "PENDING REVIEW",
    interlockLocked: false,
  };

  const strokeDashoffset = 440 - (440 * risk) / 100;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
      <div className="grid gap-6 lg:grid-cols-12 items-center">
        {/* Left: Minimalist Radial Dial */}
        <div className="lg:col-span-4 flex flex-col items-center justify-center p-2 border-b lg:border-b-0 lg:border-r border-white/[0.06] pb-6 lg:pb-0 lg:pr-6">
          <div className="relative flex items-center justify-center">
            <svg className="h-40 w-40 -rotate-90 transform" viewBox="0 0 160 160">
              <circle
                cx="80"
                cy="80"
                r="70"
                className="text-white/[0.06]"
                strokeWidth="8"
                stroke="currentColor"
                fill="transparent"
              />
              <circle
                cx="80"
                cy="80"
                r="70"
                stroke={verdictConfig.dialStroke}
                strokeWidth="8"
                strokeDasharray={440}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="transparent"
                className="transition-all duration-700 ease-out"
              />
            </svg>

            <div className="absolute flex flex-col items-center justify-center text-center">
              <span className="font-mono text-3xl font-semibold tracking-tight text-white">
                {risk.toFixed(1)}%
              </span>
              <span className="font-mono text-[9.5px] uppercase tracking-wider text-[#888888] mt-0.5">
                THREAT SCORE
              </span>
            </div>
          </div>

          <div className="mt-3 text-center">
            <span
              className={`inline-flex items-center gap-1 rounded-full border px-3 py-0.5 font-mono text-[11px] font-medium tracking-wide uppercase ${verdictConfig.badge}`}
            >
              {report.verdict}
            </span>
          </div>
        </div>

        {/* Right: Technical Evidence Telemetry */}
        <div className="lg:col-span-8 flex flex-col justify-between">
          <div>
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-md bg-[#171717] border border-white/[0.08]">
                  {verdictConfig.icon}
                </div>
                <div>
                  <h3 className="text-xl font-semibold tracking-[-0.96px] text-white">
                    {verdictConfig.title}
                  </h3>
                  <p className="font-mono text-xs text-[#a1a1a1]">
                    {verdictConfig.subtitle}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={onDownloadReport}
                className="hidden sm:flex items-center gap-1.5 rounded-full border border-white/[0.12] bg-white text-black px-4 py-1.5 text-xs font-medium hover:bg-neutral-200 transition cursor-pointer"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Audit certificate</span>
              </button>
            </div>

            <p className="mt-3.5 text-xs leading-relaxed text-[#a1a1a1] bg-[#121212] p-3.5 rounded-md border border-white/[0.06]">
              {verdictConfig.description}
            </p>

            {/* 4-Vector Evidence Metrics Bar */}
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs">
              <div className="rounded-md border border-white/[0.06] bg-[#141414] p-3">
                <span className="text-[10px] uppercase text-[#888888] block">TARGET ANCHOR ASV</span>
                <span className="text-sm font-semibold text-white mt-1 block">
                  {report.asv_evaluated === false
                    ? "N/A"
                    : report.mean_speaker_sim !== null && report.mean_speaker_sim !== undefined
                    ? `${(report.mean_speaker_sim * 100).toFixed(1)}%`
                    : "N/A"}
                </span>
                <span className="text-[10px] text-[#a1a1a1] block mt-0.5">
                  {report.asv_evaluated === false
                    ? "Not evaluated — no claimed identity"
                    : report.mean_speaker_sim && report.mean_speaker_sim >= 0.55
                    ? "Target match"
                    : "Mismatch"}
                </span>
              </div>

              <div className="rounded-md border border-white/[0.06] bg-[#141414] p-3">
                <span className="text-[10px] uppercase text-[#888888] block">VOCODER FLATNESS</span>
                <span className="text-sm font-semibold text-white mt-1 block">
                  {report.mean_high_band_flatness !== null && report.mean_high_band_flatness !== undefined
                    ? report.mean_high_band_flatness.toFixed(4)
                    : "N/A"}
                </span>
                <span className="text-[10px] text-[#a1a1a1] block mt-0.5">
                  {report.mean_high_band_flatness && report.mean_high_band_flatness >= 0.55 ? "Vocoder artifact" : "Natural"}
                </span>
              </div>

              <div className="rounded-md border border-white/[0.06] bg-[#141414] p-3">
                <span className="text-[10px] uppercase text-[#888888] block">8–14 HZ TREMOR</span>
                <span className="text-sm font-semibold text-white mt-1 block">
                  {report.mean_micro_tremor !== null && report.mean_micro_tremor !== undefined
                    ? report.mean_micro_tremor.toFixed(4)
                    : "N/A"}
                </span>
                <span className="text-[10px] text-[#a1a1a1] block mt-0.5">Lippold band</span>
              </div>

              <div className="rounded-md border border-white/[0.06] bg-[#141414] p-3">
                <span className="text-[10px] uppercase text-[#888888] block">PHASE JUMPS</span>
                <span className="text-sm font-semibold text-white mt-1 block">
                  {report.vocoder_phase_discontinuities_detected ?? 0}
                </span>
                <span className="text-[10px] text-[#a1a1a1] block mt-0.5">
                  {report.vocoder_phase_discontinuities_detected ? "Splice disruption" : "Continuous"}
                </span>
              </div>
            </div>
          </div>

          {/* Enrolled-voice identity check — the trained ml/ model.
              Reported separately from the risk tier on purpose: it answers
              "is this the enrolled speaker?", never "is this a clone?". */}
          {report.target_speaker && (
            <div className="mt-3 rounded-md border border-white/[0.06] bg-[#121212] p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  {report.target_speaker.available ? (
                    report.target_speaker.is_target ? (
                      <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
                    ) : (
                      <UserX className="h-4 w-4 text-rose-400 shrink-0" />
                    )
                  ) : (
                    <Cpu className="h-4 w-4 text-[#888888] shrink-0" />
                  )}
                  <div>
                    <span className="text-[#a1a1a1]">Enrolled-voice identity check: </span>
                    <span
                      className={`font-mono font-medium ${
                        !report.target_speaker.available
                          ? "text-[#888888]"
                          : report.target_speaker.is_target
                          ? "text-emerald-400"
                          : "text-rose-400"
                      }`}
                    >
                      {report.target_speaker.available
                        ? report.target_speaker.is_target
                          ? "IDENTITY MATCH — not a clone check"
                          : "NO MATCH — not the enrolled voice"
                        : "UNAVAILABLE"}
                    </span>
                  </div>
                </div>

                {report.target_speaker.available && report.target_speaker.prob_target !== null && report.target_speaker.prob_target !== undefined && (
                  <div className="font-mono text-[10.5px] text-[#888888]">
                    P(target)=<span className="text-[#a1a1a1]">{(report.target_speaker.prob_target * 100).toFixed(1)}%</span>
                    {" · "}LLR=<span className="text-[#a1a1a1]">{report.target_speaker.llr?.toFixed(3)}</span>
                  </div>
                )}
              </div>

              <p className="mt-1.5 text-[10.5px] leading-relaxed text-[#666666]">
                {report.target_speaker.available
                  ? "Answers \"is this the enrolled person's voice?\" — NOT \"is this a clone of them?\". The model was trained on one speaker with zero cloned audio, so a good synthetic clone of the enrolled voice would also score IDENTITY MATCH. Clone detection is the separate CM layer; train it on a real spoof corpus (ml/dataset.py::ASVspoofDataset) before relying on it."
                  : (report.target_speaker.reason || "Trained identity model not loaded.")}
              </p>
            </div>
          )}

          {/* Live transcript + scam-keyword flags — the parallel ASR/keyword
              context signal (vanirakshak/semantic/). Advisory: it nudges the
              acoustic risk fusion by a small additive term, but the acoustic
              pipeline stays the sole source of `verdict` above. */}
          {report.semantic && (
            <div className="mt-3 rounded-md border border-white/[0.06] bg-[#121212] p-3">
              <div className="flex items-center gap-2">
                <MessageSquareWarning
                  className={`h-4 w-4 shrink-0 ${
                    report.semantic.available && (report.semantic.categories_hit?.length ?? 0) > 0
                      ? "text-amber-400"
                      : "text-[#888888]"
                  }`}
                />
                <span className="text-[#a1a1a1]">Live transcript &amp; claim check: </span>
                <span
                  className={`font-mono font-medium ${
                    !report.semantic.available
                      ? "text-[#888888]"
                      : (report.semantic.claim_findings?.length ?? 0) > 0
                      ? "text-rose-400"
                      : (report.semantic.categories_hit?.length ?? 0) > 0
                      ? "text-amber-400"
                      : "text-emerald-400"
                  }`}
                >
                  {!report.semantic.available
                    ? "UNAVAILABLE"
                    : (report.semantic.claim_findings?.length ?? 0) > 0
                    ? `${report.semantic.claim_findings!.length} false claim${report.semantic.claim_findings!.length === 1 ? "" : "s"} verified`
                    : (report.semantic.categories_hit?.length ?? 0) > 0
                    ? `${report.semantic.categories_hit!.length} scam-pattern categor${report.semantic.categories_hit!.length === 1 ? "y" : "ies"} flagged`
                    : "no scam vocabulary detected"}
                </span>
              </div>

              {report.semantic.available ? (
                <>
                  {(report.semantic.hinglish_text || report.semantic.transcript) && (
                    <p className="mt-1.5 text-[11px] leading-relaxed text-[#a1a1a1] bg-[#171717] p-2 rounded border border-white/[0.05] font-mono">
                      &ldquo;{report.semantic.hinglish_text || report.semantic.transcript}&rdquo;
                    </p>
                  )}
                  {(report.semantic.matches?.length ?? 0) > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {report.semantic.matches!.map((m, i) => (
                        <span
                          key={`${m.category}-${m.term}-${i}`}
                          className="rounded-full border border-amber-500/25 bg-amber-500/10 px-2 py-0.5 text-[10px] font-mono text-amber-400"
                          title={m.category.replace(/_/g, " ")}
                        >
                          {m.term}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Verified policy contradictions. Ranked above the keyword
                      chips on purpose: a chip means "this word occurred", a
                      finding means "the caller demanded something that
                      contradicts a published rule", and names the rule. */}
                  {(report.semantic.claim_findings?.length ?? 0) > 0 && (
                    <div className="mt-2.5 space-y-1.5">
                      <div className="flex items-center gap-1.5">
                        <ShieldAlert className="h-3.5 w-3.5 text-rose-400 shrink-0" />
                        <span className="font-mono text-[10px] uppercase tracking-wider text-rose-400">
                          Verified false claims ({report.semantic.claim_findings!.length})
                        </span>
                      </div>
                      {report.semantic.claim_findings!.map((f) => (
                        <div
                          key={f.rule_id}
                          className="rounded-md border border-rose-500/25 bg-rose-500/[0.07] p-2.5"
                        >
                          <div className="text-[11px] font-medium text-rose-300">{f.claim}</div>
                          <p className="mt-1 text-[10.5px] leading-relaxed text-[#a1a1a1]">{f.finding}</p>
                          <div className="mt-1 font-mono text-[9.5px] text-[#666666]">
                            Source: {f.authority}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <p className="mt-1.5 text-[10.5px] leading-relaxed text-[#666666]">
                  {report.semantic.reason || "ASR/keyword signal not loaded."}
                </p>
              )}
              <p className="mt-1.5 text-[10.5px] leading-relaxed text-[#666666]">
                Context signal only — feeds a small term into the risk fusion, never the primary evidence.
                Claim checks verify demands against published policy (e.g. no bank ever asks for an OTP);
                they cannot verify case-specific facts like a balance or branch name.
              </p>
            </div>
          )}

          {/* Autonomous Pre-Transaction Interlock Ribbon */}
          <div className="mt-4 rounded-md border border-white/[0.06] bg-[#121212] p-3 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              {verdictConfig.interlockLocked ? (
                <Lock className="h-4 w-4 text-rose-400 shrink-0" />
              ) : (
                <Unlock className="h-4 w-4 text-emerald-400 shrink-0" />
              )}
              <div>
                <span className="text-[#a1a1a1]">Gateway interlock: </span>
                <span
                  className={`font-mono font-medium ${
                    verdictConfig.interlockLocked ? "text-rose-400" : "text-emerald-400"
                  }`}
                >
                  {verdictConfig.interlockStatus}
                </span>
              </div>
            </div>

            <div className="text-[10.5px] font-mono text-[#888888]">
              Receipt SHA-256: <span className="text-[#a1a1a1]">{report.dpdp_compliance?.receipt_sha256?.slice(0, 16)}...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
