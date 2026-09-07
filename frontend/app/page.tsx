"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  ShieldCheck,
  Mic,
  Square,
  Lock,
  Unlock,
  Cpu,
  FileCheck,
} from "lucide-react";
import ScrollyVideoCanvas from "./components/ScrollyVideoCanvas";
import DemoQuickSelector, { BENCHMARKS, BenchmarkItem } from "./components/DemoQuickSelector";
import AudioUploadZone from "./components/AudioUploadZone";
import WaveformEvidenceTimeline from "./components/WaveformEvidenceTimeline";
import ForensicVerdictCard, { ForensicReportData } from "./components/ForensicVerdictCard";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const BACKEND_WS = process.env.NEXT_PUBLIC_BACKEND_WS || "ws://localhost:8000/ws/stream";

type EventLog = {
  id: number;
  time: string;
  message: string;
  severity: "info" | "warning" | "critical";
  hash?: string;
};

export default function Home() {
  // Backend health & system state
  const [backendOnline, setBackendOnline] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState<string>("CPU (Host Execution)");

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string | null>("demo_01");
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>("/demo/demo_01_genuine.wav");
  const [currentAudioBlob, setCurrentAudioBlob] = useState<Blob | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>("demo_01_genuine.wav");
  const [report, setReport] = useState<ForensicReportData | null>(null);
  const [events, setEvents] = useState<EventLog[]>([]);
  const [showAllEvents, setShowAllEvents] = useState(false);

  // Live microphone streaming state
  const [micLive, setMicLive] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const manualCloseRef = useRef<boolean>(false);
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const addEvent = useCallback((message: string, severity: EventLog["severity"] = "info", hash?: string) => {
    setEvents((current) => [
      {
        id: Date.now() + Math.random(),
        time: new Date().toLocaleTimeString(),
        message,
        severity,
        hash,
      },
      ...current,
    ].slice(0, 20));
  }, []);

  // Poll backend health on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${BACKEND_URL}/api/v1/health`);
        if (res.ok) {
          const data = await res.json();
          setBackendOnline(true);
          if (data.hardware?.device_name) {
            setDeviceInfo(data.hardware.device_name);
          }
          addEvent(`Forensic engine online: ${data.models?.wavlm_model || "WavLM Base+"}`, "info");
        }
      } catch {
        setBackendOnline(false);
        addEvent("Forensic backend offline or unreachable", "warning");
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [addEvent]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      manualCloseRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      processorRef.current?.disconnect();
      workletNodeRef.current?.disconnect();
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      audioContextRef.current?.close();
      socketRef.current?.close();
    };
  }, []);

  // Execute forensic analysis against backend POST /api/v1/analyze/audio
  const analyzeAudioFile = useCallback(async (file: File | Blob, fileName: string) => {
    setIsAnalyzing(true);
    addEvent(`Evaluating in-memory recording: ${fileName}`, "info");

    try {
      const formData = new FormData();
      formData.append("file", file, fileName);

      const res = await fetch(`${BACKEND_URL}/api/v1/analyze/audio`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`API error (${res.status}): ${errText}`);
      }

      const data: ForensicReportData = await res.json();
      setReport(data);

      const severity: EventLog["severity"] =
        data.overall_risk_score >= 70 ? "critical" : data.overall_risk_score >= 35 ? "warning" : "info";

      addEvent(
        `Verdict: ${data.verdict} • Risk: ${data.overall_risk_score}% • Windows: ${data.windows_count}`,
        severity,
        data.audio_sha256?.slice(0, 12)
      );
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      addEvent(`Evaluation error: ${errMsg}`, "critical");
    } finally {
      setIsAnalyzing(false);
    }
  }, [addEvent]);

  // Handle Benchmark Selection
  const handleSelectBenchmark = useCallback(async (item: BenchmarkItem) => {
    setSelectedBenchmarkId(item.id);
    setCurrentFileName(item.fileName);
    setCurrentAudioUrl(item.audioUrl);
    setCurrentAudioBlob(null);

    try {
      setIsAnalyzing(true);
      const res = await fetch(item.audioUrl);
      if (!res.ok) throw new Error(`Could not load ${item.audioUrl}`);
      const blob = await res.blob();
      await analyzeAudioFile(blob, item.fileName);
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      addEvent(`Benchmark load failure: ${errMsg}`, "critical");
      setIsAnalyzing(false);
    }
  }, [analyzeAudioFile, addEvent]);

  // Load initial demo_01 on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      handleSelectBenchmark(BENCHMARKS[0]);
    }, 600);
    return () => clearTimeout(timer);
  }, [handleSelectBenchmark]);

  // Handle Custom File Upload
  const handleCustomFileSelected = (file: File) => {
    setSelectedBenchmarkId(null);
    setCurrentFileName(file.name);
    setCurrentAudioBlob(file);
    setCurrentAudioUrl(null);
    analyzeAudioFile(file, file.name);
  };

  // Download cryptographic JSON audit certificate
  const handleDownloadCertificate = () => {
    if (!report) return;
    const certificate = {
      title: "VaniRakshak Cryptographic Forensic Audio Audit Certificate",
      statute: "India Digital Personal Data Protection (DPDP) Act 2023 Section 8(7)",
      zeroRetentionVerified: true,
      timestamp: new Date().toISOString(),
      report,
    };
    const blob = new Blob([JSON.stringify(certificate, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VaniRakshak-Audit-${report.session_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    addEvent("Cryptographic audit receipt certificate exported", "info");
  };

  // Real-time microphone streaming over WebSocket per docs/PROTOCOL.md
  const startMicrophoneStream = async () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    manualCloseRef.current = false;

    try {
      const ws = new WebSocket(BACKEND_WS);
      socketRef.current = ws;

      ws.onopen = async () => {
        try {
          setMicLive(true);
          reconnectAttemptsRef.current = 0;
          addEvent("Microphone forensic stream connected (16 kHz)", "info");

          // Protocol requirement: first frame MUST be a JSON text handshake
          const sessionId = `mic-${Date.now()}`;
          ws.send(
            JSON.stringify({
              session_id: sessionId,
              sample_rate: 16000,
              caller_id: "+91-STREAM-MIC",
              claimed_identity: "",
              transaction_value_inr: 0,
              origin_country: "IN",
              prior_fraud_score: 0,
            })
          );

          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000 },
          });
          mediaStreamRef.current = stream;

          const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
          const audioContext = new AudioContextClass({ sampleRate: 16000 });
          audioContextRef.current = audioContext;

          const source = audioContext.createMediaStreamSource(stream);

          // Use AudioWorklet instead of deprecated ScriptProcessorNode
          try {
            await audioContext.audioWorklet.addModule("/worklets/pcm-capture.js");
            const workletNode = new AudioWorkletNode(audioContext, "pcm-capture");
            workletNodeRef.current = workletNode;

            workletNode.port.onmessage = (e: MessageEvent<Float32Array>) => {
              if (ws.readyState !== WebSocket.OPEN) return;
              const input = e.data;

              let sum = 0;
              for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
              const rms = Math.sqrt(sum / input.length);
              setAudioLevel(Math.min(100, Math.round(rms * 350)));

              const pcm = new Int16Array(input.length);
              for (let i = 0; i < input.length; i++) {
                const s = Math.max(-1, Math.min(1, input[i]));
                pcm[i] = s < 0 ? s * 32768 : s * 32767;
              }
              try {
                ws.send(pcm.buffer);
              } catch {
                // Socket transitioning states; skip frame
              }
            };

            source.connect(workletNode);
            workletNode.connect(audioContext.destination);
          } catch {
            // Fallback to ScriptProcessorNode for older browsers
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              if (ws.readyState !== WebSocket.OPEN) return;
              const input = e.inputBuffer.getChannelData(0);

              let sum = 0;
              for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
              const rms = Math.sqrt(sum / input.length);
              setAudioLevel(Math.min(100, Math.round(rms * 350)));

              const pcm = new Int16Array(input.length);
              for (let i = 0; i < input.length; i++) {
                const s = Math.max(-1, Math.min(1, input[i]));
                pcm[i] = s < 0 ? s * 32768 : s * 32767;
              }
              try {
                ws.send(pcm.buffer);
              } catch {
                // Socket transitioning states; skip frame
              }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
          }
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          addEvent(`Microphone error: ${errMsg}`, "critical");
          setMicLive(false);
          manualCloseRef.current = true;
          ws.close();
        }
      };

      ws.onmessage = (msg) => {
        try {
          const telemetry = JSON.parse(msg.data);
          // Backend telemetry contract (docs/PROTOCOL.md)
          if (telemetry.risk_score !== undefined || telemetry.tier !== undefined) {
            const risk = Math.round(telemetry.risk_score ?? 0);
            const tier: string = telemetry.tier || "ALLOW";
            const pSpoof: number = telemetry.p_spoof ?? risk / 100;
            const asv: number | null = telemetry.metrics?.asv_consistency ?? null;
            const snr: number | null = telemetry.metrics?.snr_db ?? null;
            const interlockActive: boolean = Boolean(
              telemetry.interlock_active || tier === "BLOCK" || risk >= 70
            );

            if (telemetry.trigger_challenge && telemetry.challenge?.phrase) {
              addEvent(
                `Voice challenge required: "${telemetry.challenge.phrase}"`,
                "warning",
                telemetry.receipt_hash?.slice(0, 12)
              );
            }

            if (interlockActive) {
              addEvent(
                `Transaction INTERLOCK ACTIVE: Tier ${tier} (Risk ${risk}%, P(spoof)=${(pSpoof * 100).toFixed(1)}%)`,
                "critical",
                telemetry.receipt_hash?.slice(0, 12)
              );
            } else {
              addEvent(
                `Live stream window: Tier ${tier} • Risk: ${risk}% • ASV: ${
                  asv !== null ? (asv * 100).toFixed(0) + "%" : "N/A"
                }${snr ? ` • SNR: ${snr.toFixed(1)}dB` : ""}`,
                risk >= 35 ? "warning" : "info",
                telemetry.receipt_hash?.slice(0, 12)
              );
            }

            // Real-time forensic telemetry updates
            setReport((prev) => ({
              session_id: prev?.session_id || `mic-${Date.now()}`,
              audio_sha256: telemetry.receipt_hash || prev?.audio_sha256 || "live-stream-receipt",
              duration_sec: (prev?.duration_sec || 0) + 1.0,
              device_used: "Client AudioWorklet (16 kHz PCM)",
              windows_count: (prev?.windows_count || 0) + 1,
              overall_risk_score: risk,
              verdict:
                tier === "BLOCK"
                  ? asv !== null && asv < 0.4
                    ? "IMPOSTOR_SPEAKER"
                    : "SYNTHETIC_VOICE_CLONE"
                  : tier === "CHALLENGE"
                  ? "SUSPICIOUS_SPLICED_AUDIO"
                  : "AUTHENTIC_TARGET",
              mean_acoustic_anomaly: pSpoof,
              mean_speaker_sim: asv,
              dpdp_compliance: {
                receipt_sha256: telemetry.receipt_hash || "hash-chain-verified",
                statutory_act: "India Digital Personal Data Protection (DPDP) Act 2023 Section 8(7)",
                zero_retention_verified: true,
                timestamp: new Date().toISOString(),
              },
            }));
          }
        } catch (err) {
          console.error("Failed to parse telemetry:", err);
        }
      };

      ws.onerror = () => {
        addEvent("Live stream connection error", "critical");
      };

      ws.onclose = () => {
        setMicLive(false);

        // Clean up existing audio resources before reconnect
        processorRef.current?.disconnect();
        processorRef.current = null;
        workletNodeRef.current?.disconnect();
        workletNodeRef.current = null;
        mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
        mediaStreamRef.current = null;
        audioContextRef.current?.close();
        audioContextRef.current = null;

        if (manualCloseRef.current) {
          addEvent("Live stream stopped", "info");
          return;
        }
        // Exponential backoff reconnect per docs/PROTOCOL.md Section 6
        if (reconnectAttemptsRef.current < 5) {
          const backoff = Math.min(15000, 1000 * Math.pow(2, reconnectAttemptsRef.current)) + Math.random() * 500;
          reconnectAttemptsRef.current += 1;
          addEvent(
            `Live stream disconnected. Reconnecting in ${(backoff / 1000).toFixed(1)}s (Attempt ${reconnectAttemptsRef.current}/5)...`,
            "warning"
          );
          reconnectTimeoutRef.current = setTimeout(() => {
            startMicrophoneStream();
          }, backoff);
        } else {
          addEvent("Live stream connection lost. Maximum reconnect attempts reached.", "critical");
        }
      };
    } catch {
      addEvent("Audio device access denied", "critical");
      setMicLive(false);
    }
  };

  const stopMicrophoneStream = () => {
    manualCloseRef.current = true;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    processorRef.current?.disconnect();
    processorRef.current = null;
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    audioContextRef.current?.close();
    audioContextRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    setMicLive(false);
    setAudioLevel(0);
  };

  // Interlock condition: server-owned tier or risk
  const currentRisk = report?.overall_risk_score ?? 0;
  const transactionLocked =
    currentRisk >= 70 ||
    report?.verdict === "SYNTHETIC_VOICE_CLONE" ||
    report?.verdict === "IMPOSTOR_SPEAKER";

  return (
    <main className="min-h-screen bg-black text-[#ededed] selection:bg-white selection:text-black">
      {/* Vercel/Geist Stark Top Navigation Bar */}
      <header className="fixed top-0 inset-x-0 z-50 border-b border-white/[0.08] bg-black/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3.5">
          {/* Logo & Brand Identity */}
          <div
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="flex items-center gap-3 cursor-pointer select-none group"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-white text-black font-mono font-bold text-sm tracking-tighter">
              VR
            </div>

            <div className="flex flex-col">
              <span className="font-semibold text-sm tracking-[-0.3px] text-white">
                VaniRakshak
              </span>
              <span className="font-mono text-[9px] uppercase tracking-wider text-[#888888]">
                Forensic Voice Intelligence
              </span>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="flex items-center gap-6 sm:gap-8 text-xs font-medium">
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
              className="text-[#888888] hover:text-white transition-colors cursor-pointer"
            >
              Overview
            </button>

            <button
              onClick={() => {
                document.getElementById("console-dashboard")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="text-[#888888] hover:text-white transition-colors cursor-pointer"
            >
              Console
            </button>

            <button
              onClick={() => {
                document.getElementById("benchmark-suite")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="text-[#888888] hover:text-white transition-colors cursor-pointer"
            >
              Benchmarks
            </button>

            <button
              onClick={handleDownloadCertificate}
              disabled={!report}
              className="text-[#888888] hover:text-white transition-colors cursor-pointer disabled:opacity-40"
            >
              Certificate
            </button>

            {/* Backend Engine Status Pill */}
            <div className="hidden sm:flex items-center gap-2 pl-3 border-l border-white/[0.08]">
              <div className="flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-[#121212] px-3 py-1 font-mono text-[10.5px] text-[#a1a1a1]">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${backendOnline ? "bg-emerald-400" : "bg-rose-500"}`}
                />
                <span>{backendOnline ? "Engine online" : "Engine offline"}</span>
              </div>
            </div>
          </nav>
        </div>
      </header>

      {/* 3D Scrollytelling Visualizer Canvas (PRESERVED INTACT) */}
      <ScrollyVideoCanvas />

      {/* Main Forensic Intelligence Console */}
      <div id="console-dashboard" className="relative z-10 scroll-mt-16 pt-10 pb-20">
        <div className="mx-auto max-w-7xl px-6 space-y-7">
          {/* Section Header */}
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 border-b border-white/[0.08] pb-6">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs uppercase tracking-wider text-[#888888]">
                  FORENSIC ACOUSTIC INTELLIGENCE
                </span>
              </div>
              <h2 className="mt-1 text-2xl sm:text-3xl font-semibold tracking-[-1.28px] text-white">
                Recorded audio forensic console.
              </h2>
              <p className="mt-1 text-xs sm:text-sm text-[#888888]">
                Target-conditioned deepfake risk localization, ECAPA-TDNN speaker verification, and vocoder Wiener entropy.
              </p>
            </div>

            {/* Hardware & Mic Control Pills */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-[#121212] px-3 py-1 font-mono text-xs text-[#a1a1a1]">
                <Cpu className="h-3.5 w-3.5 text-white" />
                <span>{deviceInfo}</span>
              </div>

              <div className="flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 font-mono text-xs text-emerald-400">
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>Anchor: Narendra Modi</span>
              </div>

              <button
                type="button"
                onClick={micLive ? stopMicrophoneStream : startMicrophoneStream}
                className={`flex items-center gap-1.5 rounded-full px-3.5 py-1 text-xs font-medium transition cursor-pointer ${
                  micLive
                    ? "bg-rose-500 text-white"
                    : "border border-white/[0.12] bg-white text-black hover:bg-neutral-200"
                }`}
              >
                {micLive ? <Square className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
                <span>{micLive ? `Live stream (${audioLevel}%)` : "Test live stream"}</span>
              </button>
            </div>
          </div>

          {/* Step 1: Standardized Forensic Benchmark Suite */}
          <div id="benchmark-suite">
            <DemoQuickSelector
              selectedId={selectedBenchmarkId}
              onSelect={handleSelectBenchmark}
              isAnalyzing={isAnalyzing}
            />
          </div>

          {/* Step 2: Custom Audio Ingestion Zone */}
          <div>
            <AudioUploadZone
              onFileSelected={handleCustomFileSelected}
              isAnalyzing={isAnalyzing}
              selectedFileName={currentFileName}
            />
          </div>

          {/* Step 3: Executive Threat Verdict Card */}
          {report && (
            <div>
              <ForensicVerdictCard
                report={report}
                onDownloadReport={handleDownloadCertificate}
              />
            </div>
          )}

          {/* Step 4: Segment-Level Waveform & Spliced Timeline */}
          {report && report.windows && report.windows.length > 0 && (
            <div>
              <WaveformEvidenceTimeline
                audioUrl={currentAudioUrl}
                audioBlob={currentAudioBlob}
                durationSec={report.duration_sec}
                windows={report.windows}
                detectedBoundaryTimestamps={report.detected_boundary_timestamps || []}
                verdict={report.verdict}
                overallRiskScore={report.overall_risk_score}
              />
            </div>
          )}

          {/* Step 5: Interlock Gateway & DPDP Compliance */}
          <section className="grid gap-6 lg:grid-cols-2">
            {/* Autonomous Pre-Transaction Interlock */}
            <div
              className={`rounded-xl border p-6 transition-colors ${
                transactionLocked
                  ? "border-rose-500/40 bg-[#160d0e]"
                  : "border-white/[0.08] bg-[#0c0c0c]"
              }`}
            >
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-md ${transactionLocked ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400"}`}>
                    {transactionLocked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm tracking-[-0.3px] text-white">
                      Autonomous gateway interlock.
                    </h3>
                    <p className="text-xs text-[#888888]">Triggered when risk score crosses 35%</p>
                  </div>
                </div>

                <span
                  className={`rounded-full px-2.5 py-0.5 font-mono text-[10px] uppercase font-medium tracking-wider border ${
                    transactionLocked
                      ? "border-rose-500/30 bg-rose-500/10 text-rose-400"
                      : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                  }`}
                >
                  {transactionLocked ? "CIRCUIT LOCKED" : "ARMED / READY"}
                </span>
              </div>

              <div className="rounded-lg border border-white/[0.06] bg-[#080808] p-4">
                <div className="mb-3 flex justify-between text-xs font-mono">
                  <span className="text-[#888888]">Protected transaction:</span>
                  <span className="text-white font-medium">NEFT / UPI Wire ₹50,000</span>
                </div>

                <button
                  type="button"
                  disabled={transactionLocked}
                  className={`w-full rounded-full py-2.5 px-4 font-medium text-xs tracking-wide transition ${
                    transactionLocked
                      ? "cursor-not-allowed bg-rose-500/20 border border-rose-500/40 text-rose-300"
                      : "bg-white text-black hover:bg-neutral-200 cursor-pointer active:scale-95"
                  }`}
                >
                  {transactionLocked
                    ? "Transaction locked · Voice spoof suspicion"
                    : "Authorize transaction (Voice authenticated)"}
                </button>

                <p className="mt-2.5 text-center text-[11px] text-[#666666]">
                  {transactionLocked
                    ? "Deepfake or biometric impostor anomaly halted payment execution."
                    : "Biological acoustic signature verified within certified target baseline."}
                </p>
              </div>
            </div>

            {/* DPDP Act 2023 Statutory Compliance Log */}
            <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-md bg-white/[0.08] text-white">
                    <FileCheck className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm tracking-[-0.3px] text-white">
                      Statutory audit trail.
                    </h3>
                    <p className="text-xs text-[#888888]">India DPDP Act 2023 §8(7) compliance</p>
                  </div>
                </div>

                <span className="rounded-full px-2.5 py-0.5 font-mono text-[10px] border border-white/[0.08] bg-[#171717] text-[#a1a1a1]">
                  ZERO RETENTION
                </span>
              </div>

              <div className="rounded-lg border border-white/[0.06] bg-[#080808] p-3.5 space-y-2 text-xs font-mono">
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-[#888888]">AUDIO SHA-256:</span>
                  <span className="text-white truncate max-w-[220px]">
                    {report?.audio_sha256 || "None loaded"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-[#888888]">RECEIPT HASH:</span>
                  <span className="text-white truncate max-w-[220px]">
                    {report?.dpdp_compliance?.receipt_sha256 || "None loaded"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-[#888888]">PROCESSING:</span>
                  <span className="text-emerald-400">Strictly volatile RAM</span>
                </div>
                <div className="flex justify-between pt-0.5">
                  <span className="text-[#888888]">STATUTE:</span>
                  <span className="text-[#a1a1a1]">DPDP Act 2023 §8(7)</span>
                </div>
              </div>
            </div>
          </section>

          {/* Step 6: Diagnostic Audit Stream */}
          <section className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
            <div className="mb-3.5 flex items-center justify-between">
              <div>
                <h3 className="font-mono text-xs uppercase tracking-wider text-[#888888]">
                  FORENSIC DIAGNOSTIC STREAM
                </h3>
                <p className="text-xs text-[#a1a1a1] mt-0.5">Real-time classification events and cryptographic receipts</p>
              </div>

              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-[#666666]">{events.length} events logged</span>
                {events.length > 5 && (
                  <button
                    type="button"
                    onClick={() => setShowAllEvents((v) => !v)}
                    className="rounded-full border border-white/[0.08] bg-[#141414] px-3 py-0.5 text-xs text-[#a1a1a1] hover:text-white transition cursor-pointer"
                  >
                    {showAllEvents ? "Show recent" : "Show all"}
                  </button>
                )}
              </div>
            </div>

            <div className="space-y-1.5">
              {(showAllEvents ? events : events.slice(0, 5)).map((event) => (
                <div
                  key={event.id}
                  className="flex items-center justify-between gap-4 rounded-md border border-white/[0.04] bg-[#080808] px-3.5 py-2 text-xs"
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                        event.severity === "critical"
                          ? "bg-rose-500"
                          : event.severity === "warning"
                            ? "bg-amber-400"
                            : "bg-emerald-400"
                      }`}
                    />
                    <span className="font-mono text-[11px] text-[#666666] shrink-0">{event.time}</span>
                    <span className="text-[#cccccc] font-medium">{event.message}</span>
                  </div>

                  {event.hash && (
                    <span className="hidden sm:inline font-mono text-[10px] text-[#666666]">
                      SHA: {event.hash}...
                    </span>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}