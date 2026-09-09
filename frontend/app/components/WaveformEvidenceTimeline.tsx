"use client";

import React, { useEffect, useRef, useState } from "react";
import { Play, Pause, RotateCcw, Volume2, Flag, Info } from "lucide-react";

export interface WindowResult {
  window_index: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  is_voiced: boolean;
  speech_ratio: number;
  acoustic_anomaly_score: number;
  speaker_cosine_sim?: number | null;
  speaker_match?: boolean | null;
  micro_tremor_ratio?: number | null;
  spectral_flatness_mean?: number | null;
  high_band_flatness?: number | null;
  zcr_variance?: number | null;
  vocoder_boundary_score?: number | null;
  inference_ms: number;
}

interface WaveformEvidenceTimelineProps {
  audioUrl?: string | null;
  audioBlob?: Blob | null;
  durationSec: number;
  windows: WindowResult[];
  detectedBoundaryTimestamps?: number[];
  verdict?: string;
  overallRiskScore?: number;
}

export default function WaveformEvidenceTimeline({
  audioUrl,
  audioBlob,
  durationSec,
  windows,
  detectedBoundaryTimestamps = [],
}: WaveformEvidenceTimelineProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeWindowIndex, setActiveWindowIndex] = useState<number | null>(null);
  const [selectedWindow, setSelectedWindow] = useState<WindowResult | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Per-bar peak amplitudes decoded from the actual audio (140 bars, 0..1),
  // tagged with the source they were decoded from. null = not decoded yet or
  // decode failed — the canvas falls back to the synthetic pattern so the
  // timeline still renders without audio.
  //
  // Tagging the source (rather than clearing peaks synchronously when the
  // source changes) is what keeps this out of react-hooks' cascading-render
  // rule: a stale decode is discarded by identity comparison at draw time
  // instead of by an extra render pass.
  type PeakState = { src: string | Blob; values: number[] };
  const [peaks, setPeaks] = useState<PeakState | null>(null);
  const peakSrc: string | Blob | null = audioBlob ?? audioUrl ?? null;
  const activePeaks = peaks && peaks.src === peakSrc ? peaks.values : null;

  const hasAudio = Boolean(audioUrl || audioBlob);

  // Decode the real waveform once per audio source and reduce channel 0 to
  // per-bar peaks. OfflineAudioContext is decode-only (no autoplay gesture
  // needed). Cross-origin URLs without CORS fail the fetch and keep peaks
  // null — colors and boundary markers stay data-driven regardless.
  useEffect(() => {
    let cancelled = false;
    if (!audioUrl && !audioBlob) return;
    const BAR_COUNT = 140;
    (async () => {
      try {
        const raw: ArrayBuffer = audioBlob
          ? await audioBlob.arrayBuffer()
          : await (await fetch(audioUrl as string)).arrayBuffer();
        const Ctx = window.OfflineAudioContext || window.AudioContext;
        const decoder = new Ctx(1, 1, 44100);
        const decoded = await decoder.decodeAudioData(raw.slice(0));
        if (cancelled) return;
        const ch = decoded.getChannelData(0);
        const out = new Array<number>(BAR_COUNT);
        let peak = 0;
        for (let i = 0; i < BAR_COUNT; i++) {
          const start = Math.floor((i / BAR_COUNT) * ch.length);
          const end = Math.max(start + 1, Math.floor(((i + 1) / BAR_COUNT) * ch.length));
          let m = 0;
          for (let j = start; j < end; j += 4) {
            const a = Math.abs(ch[j]);
            if (a > m) m = a;
          }
          out[i] = m;
          if (m > peak) peak = m;
        }
        if (peak > 0) for (let i = 0; i < BAR_COUNT; i++) out[i] /= peak;
        setPeaks({ src: (audioBlob ?? audioUrl) as string | Blob, values: out });
      } catch {
        if (!cancelled) setPeaks(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [audioBlob, audioUrl]);

  useEffect(() => {
    const audioElement = audioRef.current;
    if (!audioElement) return;

    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      audioElement.src = url;
      return () => {
        URL.revokeObjectURL(url);
        audioElement.removeAttribute("src");
      };
    } else if (audioUrl) {
      audioElement.src = audioUrl;
      return () => {
        audioElement.removeAttribute("src");
      };
    } else {
      audioElement.removeAttribute("src");
    }
  }, [audioBlob, audioUrl]);

  const handleTimeUpdate = () => {
    if (!audioRef.current) return;
    const time = audioRef.current.currentTime;
    setCurrentTime(time);

    const currentWindow = windows.find((w) => time >= w.start_sec && time < w.end_sec);
    if (currentWindow) {
      setActiveWindowIndex(currentWindow.window_index);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
    setActiveWindowIndex(null);
  };

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const restartAudio = () => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = 0;
    setCurrentTime(0);
    audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
  };

  const seekTo = (seconds: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = seconds;
    setCurrentTime(seconds);
    const win = windows.find((w) => seconds >= w.start_sec && seconds < w.end_sec);
    if (win) setSelectedWindow(win);
  };

  // Render forensic waveform canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const barCount = 140;
    const barWidth = width / barCount - 1.5;
    const totalDuration = durationSec || 10.0;

    // Draw subtle grid markers
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      const x = (width / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Draw waveform bars with risk coloring
    for (let i = 0; i < barCount; i++) {
      const barTime = (i / barCount) * totalDuration;
      const x = i * (barWidth + 1.5);

      const win = windows.find((w) => barTime >= w.start_sec && barTime < w.end_sec);
      
      let fillColor = "#10b981"; // emerald (bona fide)
      if (win) {
        const anomaly = win.acoustic_anomaly_score || 0;
        const flatness = win.high_band_flatness || 0;
        const sim = win.speaker_cosine_sim;

        if (sim !== null && sim !== undefined && sim < 0.55) {
          fillColor = "#a855f7"; // purple (impostor)
        } else if (flatness >= 0.55 || anomaly >= 0.60) {
          fillColor = "#f43f5e"; // rose (synthetic clone)
        } else if (win.vocoder_boundary_score && win.vocoder_boundary_score >= 0.25) {
          fillColor = "#f5a623"; // amber (spliced boundary)
        } else if (anomaly > 0.40) {
          fillColor = "#f5a623";
        }
      }

      // Real decoded peaks when available; synthetic sine fallback only
      // while decoding or when the source can't be fetched/decoded.
      let amp: number;
      if (activePeaks) {
        amp = Math.max(0.03, Math.min(1, activePeaks[i] ?? 0));
      } else {
        const baseAmp = Math.sin((i / barCount) * Math.PI * 4) * 0.3 + 0.5;
        const noise = (Math.sin(i * 12.3) * 0.5 + 0.5) * 0.4;
        amp = Math.max(0.12, Math.min(0.92, baseAmp + noise));
      }
      const barH = Math.max(1.5, amp * (height * 0.72));
      const y = (height - barH) / 2;

      const isPast = barTime <= currentTime;
      ctx.fillStyle = isPast ? fillColor : `${fillColor}55`;
      ctx.fillRect(x, y, barWidth, barH);
    }

    // Draw phase boundary pins if detected
    detectedBoundaryTimestamps.forEach((ts) => {
      const pinX = (ts / totalDuration) * width;
      ctx.strokeStyle = "#f5a623";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(pinX, 0);
      ctx.lineTo(pinX, height);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#f5a623";
      ctx.beginPath();
      ctx.arc(pinX, 8, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw current playback scrubber line
    const scrubberX = (currentTime / totalDuration) * width;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(scrubberX, 0);
    ctx.lineTo(scrubberX, height);
    ctx.stroke();
  }, [windows, durationSec, currentTime, detectedBoundaryTimestamps, activePeaks]);

  const formatSec = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = (s % 60).toFixed(1);
    return `${mins.toString().padStart(2, "0")}:${secs.padStart(4, "0")}`;
  };

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
      <audio
        ref={audioRef}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        preload="auto"
      />

      {/* Header bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4 border-b border-white/[0.06] pb-4">
        <div>
          <span className="font-mono text-xs uppercase tracking-wider text-[#a1a1a1]">
            SEGMENT-LEVEL TIMELINE
          </span>
          <h3 className="mt-1 text-lg font-semibold tracking-[-0.6px] text-white">
            Forensic waveform and splice intervals.
          </h3>
          <p className="mt-0.5 text-xs text-[#888888]">
            2.0-second sliding windows with 0.5-second stride evaluated against target speaker anchor.
          </p>
        </div>

        {/* Transport controls */}
        <div className="flex items-center gap-2.5">
          <div className="font-mono text-xs text-[#a1a1a1] bg-[#141414] px-3 py-1.5 rounded-full border border-white/[0.06] flex items-center gap-1.5">
            <Volume2 className="h-3.5 w-3.5 text-white" />
            <span className="text-white font-medium">{formatSec(currentTime)}</span>
            <span className="text-[#555] font-normal">/</span>
            <span>{formatSec(durationSec || 10.0)}</span>
          </div>

          <button
            onClick={togglePlay}
            disabled={!hasAudio}
            className="flex items-center gap-1.5 rounded-full bg-white text-black px-4 py-1.5 text-xs font-medium transition hover:bg-neutral-200 active:scale-95 cursor-pointer disabled:opacity-40"
          >
            {isPlaying ? <Pause className="h-3 w-3 fill-current" /> : <Play className="h-3 w-3 fill-current" />}
            <span>{isPlaying ? "Pause" : "Play"}</span>
          </button>

          <button
            onClick={restartAudio}
            disabled={!hasAudio}
            className="rounded-full border border-white/[0.1] bg-[#141414] p-2 text-[#a1a1a1] hover:text-white transition active:scale-95 cursor-pointer disabled:opacity-40"
            title="Restart playback"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Waveform Canvas */}
      <div className="relative rounded-lg border border-white/[0.06] bg-[#080808] p-3">
        <canvas
          ref={canvasRef}
          width={880}
          height={110}
          className="w-full h-[110px] rounded cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const clickRatio = (e.clientX - rect.left) / rect.width;
            seekTo(clickRatio * (durationSec || 10.0));
          }}
        />

        {/* Legend */}
        <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.06] pt-2 text-[10.5px] font-mono text-[#888888]">
          <div className="flex items-center gap-3.5">
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              <span>Authentic (&lt;35%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-amber-400" />
              <span>Suspicious (35–70%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-rose-500" />
              <span>Clone (&ge;70%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-purple-400" />
              <span>Impostor</span>
            </div>
          </div>

          {detectedBoundaryTimestamps.length > 0 && (
            <div className="flex items-center gap-1 text-amber-400">
              <Flag className="h-3 w-3" />
              <span>{detectedBoundaryTimestamps.length} phase jump detected: {detectedBoundaryTimestamps.map(t => `${t}s`).join(", ")}</span>
            </div>
          )}
        </div>
      </div>

      {/* Segment Windows Grid */}
      <div className="mt-5">
        <div className="flex items-center justify-between mb-2">
          <div className="font-mono text-xs uppercase text-[#888888]">
            WINDOW TELEMETRY ({windows.length} SEGMENTS)
          </div>
          <span className="text-[11px] text-[#666666]">Click any segment to seek</span>
        </div>

        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin">
          {windows.map((w) => {
            const isWindowActive = activeWindowIndex === w.window_index;
            const isWindowSelected = selectedWindow?.window_index === w.window_index;

            const isImpostor = w.speaker_cosine_sim !== null && w.speaker_cosine_sim !== undefined && w.speaker_cosine_sim < 0.55;
            const isClone = (w.high_band_flatness || 0) >= 0.55 || (w.acoustic_anomaly_score || 0) >= 0.60;
            const isSpliced = w.vocoder_boundary_score && w.vocoder_boundary_score >= 0.25;

            let cardBorder = "border-emerald-500/20 bg-[#101713] text-emerald-300";
            let statusBadge = "BONA FIDE";

            if (isImpostor) {
              cardBorder = "border-purple-500/20 bg-[#16101c] text-purple-300";
              statusBadge = "IMPOSTOR";
            } else if (isClone) {
              cardBorder = "border-rose-500/20 bg-[#1a1012] text-rose-300";
              statusBadge = "CLONE";
            } else if (isSpliced) {
              cardBorder = "border-amber-500/20 bg-[#1a140f] text-amber-300";
              statusBadge = "SPLICE";
            }

            return (
              <div
                key={w.window_index}
                onClick={() => {
                  seekTo(w.start_sec);
                  setSelectedWindow(w);
                }}
                className={`flex-shrink-0 w-32 rounded-md border p-2.5 transition-all duration-150 cursor-pointer ${cardBorder} ${
                  isWindowActive ? "ring-1 ring-white" : ""
                } ${isWindowSelected ? "border-white" : ""}`}
              >
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-[#a1a1a1]">W#{w.window_index}</span>
                  <span className="rounded px-1.5 py-0.2 bg-black/60 text-[9px] font-medium">{statusBadge}</span>
                </div>

                <div className="mt-1 font-mono text-[11px] font-medium text-white">
                  {w.start_sec.toFixed(1)}s–{w.end_sec.toFixed(1)}s
                </div>

                <div className="mt-2 space-y-0.5 text-[9.5px] font-mono text-[#888888]">
                  <div className="flex justify-between">
                    <span>Anomaly:</span>
                    <span className="text-white font-medium">{(w.acoustic_anomaly_score || 0).toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>ASV:</span>
                    <span className="text-white font-medium">
                      {w.speaker_cosine_sim !== null && w.speaker_cosine_sim !== undefined
                        ? w.speaker_cosine_sim.toFixed(2)
                        : "N/A"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Flatness:</span>
                    <span className="text-white font-medium">{(w.high_band_flatness || 0).toFixed(3)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Window Detail Callout */}
      {selectedWindow && (
        <div className="mt-4 rounded-md border border-white/[0.08] bg-[#121212] p-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-2 mb-3">
            <div className="flex items-center gap-2">
              <Info className="h-4 w-4 text-[#a1a1a1]" />
              <h4 className="font-mono text-xs text-white font-medium">
                Window #{selectedWindow.window_index} telemetry [{selectedWindow.start_sec.toFixed(1)}s – {selectedWindow.end_sec.toFixed(1)}s]
              </h4>
            </div>
            <button
              onClick={() => setSelectedWindow(null)}
              className="text-[#888888] hover:text-white text-xs font-mono cursor-pointer"
            >
              Close ✕
            </button>
          </div>

          <div className="grid gap-2.5 grid-cols-2 sm:grid-cols-4 text-xs font-mono">
            <div className="rounded bg-[#171717] p-2.5 border border-white/[0.04]">
              <div className="text-[10px] text-[#888888] uppercase">WAVLM ANOMALY</div>
              <div className="text-sm font-semibold text-white mt-1">
                {(selectedWindow.acoustic_anomaly_score || 0).toFixed(4)}
              </div>
            </div>

            <div className="rounded bg-[#171717] p-2.5 border border-white/[0.04]">
              <div className="text-[10px] text-[#888888] uppercase">ECAPA ASV COSINE</div>
              <div className="text-sm font-semibold text-white mt-1">
                {selectedWindow.speaker_cosine_sim !== null && selectedWindow.speaker_cosine_sim !== undefined
                  ? selectedWindow.speaker_cosine_sim.toFixed(4)
                  : "N/A"}
              </div>
            </div>

            <div className="rounded bg-[#171717] p-2.5 border border-white/[0.04]">
              <div className="text-[10px] text-[#888888] uppercase">HIGH-BAND FLATNESS</div>
              <div className="text-sm font-semibold text-white mt-1">
                {(selectedWindow.high_band_flatness || 0).toFixed(4)}
              </div>
            </div>

            <div className="rounded bg-[#171717] p-2.5 border border-white/[0.04]">
              <div className="text-[10px] text-[#888888] uppercase">WINDOW LATENCY</div>
              <div className="text-sm font-semibold text-white mt-1">
                {selectedWindow.inference_ms.toFixed(1)} ms
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
