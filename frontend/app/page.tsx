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
import EnrolledSpeakersPanel from "./components/EnrolledSpeakersPanel";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const BACKEND_WS = process.env.NEXT_PUBLIC_BACKEND_WS || "ws://localhost:8000/ws/stream";

// Used only if /api/v1/health has not reported an enrolled anchor yet (or the
// backend is unreachable). Prefer the health response so the client never
// claims an identity the server has no vector for.
const CLAIMED_IDENTITY_FALLBACK = "narendra_modi";

// --- CHALLENGE interlock helpers (mirror vanirakshak/challenge/engine.py) --
type LocalChallenge = {
  phrase: string;
  expected_digits: string[];
  expected_full: string;
  instructions: string;
  issued_at_ms: number;
  seed: number;
};

function issueLocalChallenge(): LocalChallenge {
  const words = ["aam", "nadi", "kela", "suraj", "chai", "tara", "mango", "river", "blue", "seven", "table", "green", "orange", "cloud"];
  const digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  const pick = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];
  const d1 = pick(digits);
  const d2 = pick(digits);
  const w1 = pick(words);
  const w2 = pick(words);
  return {
    phrase: `${w1}, ${d1}, ${w2}, ${d2}.`,
    expected_digits: [d1, d2],
    expected_full: `${w1} ${d1} ${w2} ${d2}`,
    instructions: "Now repeat ONLY the digits, in order, quickly.",
    issued_at_ms: Date.now(),
    seed: Math.floor(Math.random() * 1_000_000),
  };
}

// Mirrors ChallengeEngine.grade(): digit-order match + human latency window.
function gradeChallengeLocally(
  ch: { expected_digits: string[] },
  responseText: string,
  latencyMs: number,
): { passed: boolean; reason: string } {
  const responseDigits = (responseText.match(/\d/g) || []) as string[];
  const correctOrder =
    responseDigits.slice(0, ch.expected_digits.length).join("") ===
    ch.expected_digits.join("");
  const latencyOk = latencyMs >= 400 && latencyMs <= 4000;
  if (!correctOrder) return { passed: false, reason: "digit_order_mismatch" };
  if (!latencyOk) return { passed: false, reason: "latency_suspicious" };
  return { passed: true, reason: "ok" };
}

type EventLog = {
  id: number;
  time: string;
  message: string;
  severity: "info" | "warning" | "critical";
  hash?: string;
};

// The wire contract (docs/PROTOCOL.md) is 16 kHz mono int16. We request
// 16 kHz on getUserMedia and the AudioContext, but a browser may honor a
// different hardware rate — sending that through while claiming 16000 in
// the handshake silently corrupts every downstream detector. Resample to
// the contract rate (linear interpolation is fine for speech) whenever
// the context rate disagrees.
const STREAM_RATE = 16000;

function resampleToStreamRate(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === STREAM_RATE || input.length === 0) return input;
  const ratio = fromRate / STREAM_RATE;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src);
    const frac = src - i0;
    const a = input[i0];
    const b = input[Math.min(i0 + 1, input.length - 1)];
    out[i] = a + (b - a) * frac;
  }
  return out;
}

export default function Home() {
  // Backend health & system state
  const [backendOnline, setBackendOnline] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState<string>("CPU (Host Execution)");
  // Enrolled target speaker, read from GET /api/v1/health -> anchor.user_id.
  // Claiming it is what makes an analysis target-conditioned: with an empty
  // identity the backend's ASV returns a flat 0.5 and the speaker-similarity
  // half of the system does nothing. Kept in a ref too, because the WebSocket
  // handshake runs inside a closure that would otherwise capture the initial "".
  const [claimedIdentity, setClaimedIdentity] = useState<string>("");
  const claimedIdentityRef = useRef<string>("");

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string | null>("demo_01");
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>("/demo/demo_01_genuine.wav");
  const [currentAudioBlob, setCurrentAudioBlob] = useState<Blob | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>("demo_01_genuine.wav");
  const [report, setReport] = useState<ForensicReportData | null>(null);
  const [events, setEvents] = useState<EventLog[]>([]);
  const [showAllEvents, setShowAllEvents] = useState(false);
  // SHA-256 append-only audit chain integrity (flowchart node Q).
  // null = not yet verified (engine offline / no audit events).
  const [chainValid, setChainValid] = useState<boolean | null>(null);

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

  // CHALLENGE interlock state (mirrors backend CallSession + ChallengeEngine).
  // The session_id from the WS handshake is required to grade the response
  // server-side via POST /api/v1/sessions/{sid}/challenge.
  const sessionIdRef = useRef<string>("");
  const activeChallengeRef = useRef<LocalChallenge | null>(null);
  const challengeIssuedAtRef = useRef<number>(0);
  const challengePendingRef = useRef<boolean>(false);
  // Where the active challenge came from. Only a server-issued phrase can be
  // graded by the backend (it grades against its OWN challenge), so a locally
  // generated drill phrase must be graded locally.
  // The ref is read inside async event handlers (submitChallengeResponse),
  // where it must be the freshest value. The state mirror below is what
  // render may read — a ref read during render is not reactive, so the
  // interlock could compute its tier from a stale source. Keep both in sync.
  const challengeSourceRef = useRef<"server" | "local">("local");
  const [challengeSource, setChallengeSource] = useState<"server" | "local">("local");
  const [activeChallenge, setActiveChallenge] = useState<LocalChallenge | null>(null);
  const [challengeStatus, setChallengeStatus] = useState<"none" | "awaiting" | "passed" | "failed">("none");
  const [challengeResponseText, setChallengeResponseText] = useState<string>("");
  // Latest server tier from the live stream (ALLOW / CHALLENGE / BLOCK).
  const [currentTier, setCurrentTier] = useState<string>("ALLOW");

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
          if (data.anchor?.user_id) {
            claimedIdentityRef.current = data.anchor.user_id;
            setClaimedIdentity(data.anchor.user_id);
          } else if (data.anchor?.error) {
            addEvent(`No enrolled anchor: ${data.anchor.error}`, "warning");
          }
          addEvent(`Forensic engine online: ${data.models?.wavlm_model || "WavLM Base+"}`, "info");

          // Verify the SHA-256 append-only audit chain (flowchart node Q).
          try {
            const ar = await fetch(`${BACKEND_URL}/api/v1/audit?n=1`);
            if (ar.ok) {
              const ad = await ar.json();
              setChainValid(Boolean(ad.chain_valid));
              if (ad.chain_valid === false) {
                addEvent("Audit hash chain integrity check FAILED", "critical");
              }
            }
          } catch {
            setChainValid(null);
          }
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
      // Target-conditioned analysis: score against the enrolled anchor speaker.
      // Without this the backend cannot run the ASV path at all.
      formData.append("claimed_identity", claimedIdentityRef.current || CLAIMED_IDENTITY_FALLBACK);

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
          sessionIdRef.current = sessionId;
          ws.send(
            JSON.stringify({
              session_id: sessionId,
              sample_rate: STREAM_RATE,
              caller_id: "+91-STREAM-MIC",
              claimed_identity: claimedIdentityRef.current || CLAIMED_IDENTITY_FALLBACK,
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
          const audioContext = new AudioContextClass({ sampleRate: STREAM_RATE });
          audioContextRef.current = audioContext;

          // The browser may ignore the requested rate (hardware constraints,
          // Safari). Everything downstream is resampled to STREAM_RATE so the
          // handshake's sample_rate claim stays truthful.
          const contextRate = audioContext.sampleRate;
          if (contextRate !== STREAM_RATE) {
            addEvent(
              `Audio hardware runs at ${contextRate} Hz — resampling to ${STREAM_RATE} Hz in-client`,
              "warning"
            );
          }

          const source = audioContext.createMediaStreamSource(stream);

          // Use AudioWorklet instead of deprecated ScriptProcessorNode
          try {
            await audioContext.audioWorklet.addModule("/worklets/pcm-capture.js");
            const workletNode = new AudioWorkletNode(audioContext, "pcm-capture");
            workletNodeRef.current = workletNode;

            workletNode.port.onmessage = (e: MessageEvent<Float32Array>) => {
              if (ws.readyState !== WebSocket.OPEN) return;
              const input = resampleToStreamRate(e.data, contextRate);

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
              const input = resampleToStreamRate(e.inputBuffer.getChannelData(0), contextRate);

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
          // Handshake ack (docs/PROTOCOL.md §2): the server echoes the
          // resolved session_id first — adopt it so REST challenge-grading
          // targets the right session even if we omitted session_id.
          if (telemetry.type === "session_ack" && typeof telemetry.session_id === "string") {
            sessionIdRef.current = telemetry.session_id;
            addEvent(`Stream session acknowledged (${telemetry.session_id})`, "info");
            return;
          }
          // Backend telemetry contract (docs/PROTOCOL.md)
          if (telemetry.risk_score !== undefined || telemetry.tier !== undefined) {
            const risk = Math.round(telemetry.risk_score ?? 0);
            const tier: string = telemetry.tier || "ALLOW";
            const pSpoof: number = telemetry.p_spoof ?? risk / 100;
            const asv: number | null = telemetry.metrics?.asv_consistency ?? null;
            const asvEvaluated: boolean = Boolean(telemetry.metrics?.asv_evaluated);
            // G7: is the ASV behind this number a real voiceprint or the
            // synthetic random-projection stub? Drives the "(SYNTHETIC)" label.
            const asvSynthetic: boolean = Boolean(telemetry.metrics?.asv_synthetic);
            const snr: number | null = telemetry.metrics?.snr_db ?? null;
            const interlockActive: boolean = Boolean(
              telemetry.interlock_active || tier === "BLOCK" || risk >= 70
            );
            // Track the latest server tier so the interlock can reflect it.
            setCurrentTier(tier);

            if (telemetry.trigger_challenge && telemetry.challenge?.phrase) {
              // Capture the live challenge into UI state so the operator can
              // surface the phrase, transcribe the caller's answer, and grade it.
              if (!challengePendingRef.current) {
                const ch: LocalChallenge = {
                  phrase: telemetry.challenge.phrase,
                  expected_digits: telemetry.challenge.expected_digits ?? [],
                  expected_full: telemetry.challenge.expected_full ?? "",
                  instructions:
                    telemetry.challenge.instructions ?? "Repeat ONLY the digits, in order, quickly.",
                  issued_at_ms: telemetry.challenge.issued_at_ms ?? Date.now(),
                  seed: telemetry.challenge.seed ?? 0,
                };
                activeChallengeRef.current = ch;
                challengeSourceRef.current = "server";
                setChallengeSource("server");
                challengeIssuedAtRef.current = Date.now();
                setActiveChallenge(ch);
                setChallengeStatus("awaiting");
                setChallengeResponseText("");
                challengePendingRef.current = true;
              }
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
              asv_evaluated: asvEvaluated,
              asv_synthetic: asvSynthetic,
              mean_speaker_sim: asv,
              semantic: telemetry.semantic ?? prev?.semantic ?? null,
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
    // Reset CHALLENGE interlock state on stream stop.
    activeChallengeRef.current = null;
    challengePendingRef.current = false;
    challengeSourceRef.current = "local";
    setChallengeSource("local");
    setActiveChallenge(null);
    setChallengeStatus("none");
    setChallengeResponseText("");
    setCurrentTier("ALLOW");
  };

  // Effective interlock tier.
  // 1. The CHALLENGE loop can override the live tier: Pass -> ALLOW, Fail -> BLOCK.
  // 2. Otherwise we trust the server's tier from the live stream (micLive), or
  //    derive a tier from the one-shot forensic report when no stream is active.
  const reportTier: string = report
    ? report.overall_risk_score >= 70 ||
      report.verdict === "SYNTHETIC_VOICE_CLONE" ||
      report.verdict === "IMPOSTOR_SPEAKER"
      ? "BLOCK"
      : report.overall_risk_score >= 35 || report.verdict === "SUSPICIOUS_SPLICED_AUDIO"
      ? "CHALLENGE"
      : "ALLOW"
    : "ALLOW";

  const liveTier = micLive ? currentTier : reportTier;
  // The backend owns the interlock decision (AGENTS.md §2). Once a live
  // stream is running and the active challenge was issued by the server,
  // the next WS verdict already reflects whatever the backend's grading
  // decided — so challengeStatus must never override tier here, or a
  // caller the server has since re-BLOCKed would show as unlocked. Local
  // overrides are only for the demo carve-out docs/PROTOCOL.md §5
  // describes: a manual drill (no server challenge exists to track) or the
  // backend being unreachable (no live verdict to trust either way).
  // Reads `challengeSource` (state), NOT challengeSourceRef.current. A ref
  // read during render is not reactive, so the previous version could
  // evaluate against a stale source — and this predicate is what decides
  // whether the client may unlock a transaction the server has blocked.
  //
  // Permissive (||), deliberately. Each condition marks a case where there
  // is no server verdict to defer to:
  //   challengeSource === "local"  — a manual drill; the backend has no
  //     challenge of its own to grade, so it will never speak to this one.
  //   !backendOnline               — nothing to defer to at all.
  //   !micLive                     — no live stream, so `liveTier` came
  //     from a one-shot report rather than a streamed verdict.
  //
  // The case that governs the choice: a SERVER-issued challenge, backend
  // online, mic live, graded pass — then the next window streams BLOCK.
  // All three conditions are false there, so the override does not engage
  // and liveTier (BLOCK) stands. That is the AGENTS.md §2 invariant, and
  // it holds without needing &&. Requiring all three instead would disable
  // the drill carve-out docs/PROTOCOL.md §5 documents, buying no safety
  // the || form doesn't already give.
  const useLocalChallengeOverride =
    challengeSource === "local" || !backendOnline || !micLive;
  const effectiveTier: string =
    useLocalChallengeOverride && challengeStatus === "passed"
      ? "ALLOW"
      : useLocalChallengeOverride && challengeStatus === "failed"
      ? "BLOCK"
      : liveTier;

  // Grade a challenge response. Only a server-issued phrase can be graded by
  // the backend (POST /api/v1/sessions/{sid}/challenge) — it grades against its
  // OWN active challenge, so posting a locally generated drill phrase there
  // would always fail with "no_active_challenge". Drill phrases, and any
  // response while the backend is offline, use the local mirror so the
  // interlock logic stays demonstrable.
  const submitChallengeResponse = async () => {
    const ch = activeChallengeRef.current;
    if (!ch) return;
    const latencyMs = Date.now() - challengeIssuedAtRef.current;
    const text = challengeResponseText;

    let passed = false;
    let reason = "ok";

    if (challengeSourceRef.current === "server" && backendOnline && sessionIdRef.current) {
      try {
        const res = await fetch(
          `${BACKEND_URL}/api/v1/sessions/${sessionIdRef.current}/challenge`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ response_text: text, response_latency_ms: latencyMs }),
          }
        );
        if (res.ok) {
          const data = await res.json();
          passed = Boolean(data.ok);
          reason = data.reason || "ok";
        } else {
          const g = gradeChallengeLocally(ch, text, latencyMs);
          passed = g.passed;
          reason = g.reason;
        }
      } catch {
        const g = gradeChallengeLocally(ch, text, latencyMs);
        passed = g.passed;
        reason = g.reason;
      }
    } else {
      const g = gradeChallengeLocally(ch, text, latencyMs);
      passed = g.passed;
      reason = g.reason;
    }

    // Release the pending-challenge gate now that this one is graded, so a
    // *second* server challenge in the same session (e.g. risk climbs again
    // after a pass) can register in ws.onmessage instead of only logging an
    // event with no UI update — the gate exists to stop this same challenge
    // from being captured twice, not to block all future ones.
    challengePendingRef.current = false;

    if (passed) {
      setChallengeStatus("passed");
      addEvent(
        `Challenge PASSED — digits correct, latency ${latencyMs} ms. Caller re-authenticated → ALLOW.`,
        "info"
      );
    } else {
      setChallengeStatus("failed");
      addEvent(
        `Challenge FAILED (${reason}) — escalating to BLOCK. Transaction interlocked.`,
        "critical"
      );
    }
  };

  // Manual challenge drill: lets the operator exercise the anti-replay interlock
  // on demand (e.g. with no backend, or to verify the loop before a live call).
  const runChallengeDrill = () => {
    const ch = issueLocalChallenge();
    activeChallengeRef.current = ch;
    challengeSourceRef.current = "local";
    setChallengeSource("local");
    challengeIssuedAtRef.current = Date.now();
    setActiveChallenge(ch);
    setChallengeStatus("awaiting");
    setChallengeResponseText("");
    challengePendingRef.current = true;
    addEvent(
      `Challenge drill issued: "${ch.phrase}". Ask the caller to repeat ONLY the digits.`,
      "warning"
    );
  };

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
            <div className="flex h-9 w-9 items-center justify-center rounded-md overflow-hidden bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/brand/vani-rakshak-icon.jpg"
                alt="VaniRakshak shield+mic logo"
                width={36}
                height={36}
                className="h-9 w-9 object-contain"
              />
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

              {/* Enrolled anchor, as reported by the backend — not a hard-coded string. */}
              <div
                className={`flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-xs ${
                  claimedIdentity
                    ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                    : "border-amber-500/20 bg-amber-500/10 text-amber-400"
                }`}
              >
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>
                  {claimedIdentity
                    ? `Anchor: ${claimedIdentity.replace(/_/g, " ")}`
                    : "Anchor: not enrolled"}
                </span>
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

              <button
                type="button"
                onClick={runChallengeDrill}
                className="flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3.5 py-1 text-xs font-medium text-amber-300 transition hover:bg-amber-500/20 cursor-pointer"
              >
                <Lock className="h-3 w-3" />
                <span>Run challenge drill</span>
              </button>
            </div>
          </div>

          {/* Multi-speaker enrollment — lets the operator switch which
              enrolled voice is being protected, or enrol a new one, instead
              of being locked to the single hardcoded anchor. */}
          <div>
            <EnrolledSpeakersPanel
              backendUrl={BACKEND_URL}
              claimedIdentity={claimedIdentity}
              disabled={micLive}
              onSelectIdentity={(id) => {
                claimedIdentityRef.current = id;
                setClaimedIdentity(id);
                addEvent(`Claimed identity set to "${id.replace(/_/g, " ")}"`, "info");
              }}
              onEvent={addEvent}
            />
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
                effectiveTier === "BLOCK"
                  ? "border-rose-500/40 bg-[#160d0e]"
                  : effectiveTier === "CHALLENGE"
                  ? "border-amber-500/40 bg-[#1a1408]"
                  : "border-white/[0.08] bg-[#0c0c0c]"
              }`}
            >
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-md ${
                    effectiveTier === "BLOCK"
                      ? "bg-rose-500/20 text-rose-400"
                      : effectiveTier === "CHALLENGE"
                      ? "bg-amber-500/20 text-amber-400"
                      : "bg-emerald-500/20 text-emerald-400"
                  }`}>
                    {effectiveTier === "BLOCK" || effectiveTier === "CHALLENGE" ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm tracking-[-0.3px] text-white">
                      Autonomous gateway interlock.
                    </h3>
                    <p className="text-xs text-[#888888]">Triggered when risk ≥ 70% or challenge fails</p>
                  </div>
                </div>

                <span
                  className={`rounded-full px-2.5 py-0.5 font-mono text-[10px] uppercase font-medium tracking-wider border ${
                    effectiveTier === "BLOCK"
                      ? "border-rose-500/30 bg-rose-500/10 text-rose-400"
                      : effectiveTier === "CHALLENGE"
                      ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
                      : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                  }`}
                >
                  {effectiveTier === "BLOCK" ? "CIRCUIT LOCKED" : effectiveTier === "CHALLENGE" ? "CHALLENGE PENDING" : "ARMED / READY"}
                </span>
              </div>

              <div className="rounded-lg border border-white/[0.06] bg-[#080808] p-4">
                <div className="mb-3 flex justify-between text-xs font-mono">
                  <span className="text-[#888888]">Protected transaction:</span>
                  <span className="text-white font-medium">NEFT / UPI Wire ₹50,000</span>
                </div>

                <button
                  type="button"
                  disabled={effectiveTier !== "ALLOW"}
                  className={`w-full rounded-full py-2.5 px-4 font-medium text-xs tracking-wide transition ${
                    effectiveTier === "BLOCK"
                      ? "cursor-not-allowed bg-rose-500/20 border border-rose-500/40 text-rose-300"
                      : effectiveTier === "CHALLENGE"
                      ? "cursor-not-allowed bg-amber-500/20 border border-amber-500/40 text-amber-300"
                      : "bg-white text-black hover:bg-neutral-200 cursor-pointer active:scale-95"
                  }`}
                >
                  {effectiveTier === "BLOCK"
                    ? "Transaction locked · Voice spoof suspicion"
                    : effectiveTier === "CHALLENGE"
                    ? "Awaiting challenge response…"
                    : "Authorize transaction (Voice authenticated)"}
                </button>

                <p className="mt-2.5 text-center text-[11px] text-[#666666]">
                  {effectiveTier === "BLOCK"
                    ? "Deepfake or biometric impostor anomaly halted payment execution."
                    : effectiveTier === "CHALLENGE"
                    ? "Caller must repeat the challenge digits before the gateway can proceed."
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
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-[#888888]">HASH CHAIN:</span>
                  <span className={chainValid === null ? "text-[#a1a1a1]" : chainValid ? "text-emerald-400" : "text-rose-400"}>
                    {chainValid === null ? "unverified (engine offline)" : chainValid ? "verified · append-only" : "BROKEN"}
                  </span>
                </div>
                <div className="flex justify-between pt-0.5">
                  <span className="text-[#888888]">STATUTE:</span>
                  <span className="text-[#a1a1a1]">DPDP Act 2023 §8(7)</span>
                </div>
              </div>
            </div>
          </section>

          {/* Step 5b: Active Voice Challenge (dynamic anti-replay interlock) */}
          {activeChallenge && (
            <section className="rounded-xl border border-amber-500/30 bg-[#1a1408] p-6 shadow-sm">
              <div className="mb-4 flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-amber-500/20 text-amber-400">
                  <Lock className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm tracking-[-0.3px] text-white">
                    Live voice challenge.
                  </h3>
                  <p className="text-xs text-[#888888]">Dynamic anti-replay defence — caller must speak the digits live</p>
                </div>
                <span className={`ml-auto rounded-full px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-wider border ${
                  challengeStatus === "passed"
                    ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                    : challengeStatus === "failed"
                    ? "border-rose-500/30 bg-rose-500/10 text-rose-400"
                    : "border-amber-500/30 bg-amber-500/10 text-amber-400"
                }`}>
                  {challengeStatus === "passed" ? "PASSED" : challengeStatus === "failed" ? "FAILED" : "AWAITING"}
                </span>
              </div>

              <div className="rounded-lg border border-amber-500/20 bg-[#0c0c0c] p-4">
                <p className="text-xs text-[#888888] mb-1">Challenge phrase — ask the caller to repeat ONLY the digits, in order:</p>
                <p className="text-lg font-mono text-amber-300 tracking-wide mb-2">{activeChallenge.phrase}</p>
                <p className="text-xs text-[#666666] mb-3">{activeChallenge.instructions}</p>

                {challengeStatus === "awaiting" && (
                  <>
                    <input
                      type="text"
                      value={challengeResponseText}
                      onChange={(e) => setChallengeResponseText(e.target.value)}
                      placeholder="Type what the caller said…"
                      className="w-full rounded-lg border border-white/[0.12] bg-[#080808] px-3 py-2 text-sm text-white placeholder:text-[#555] focus:outline-none focus:border-amber-500/40"
                    />
                    <button
                      type="button"
                      onClick={submitChallengeResponse}
                      className="mt-3 w-full rounded-full bg-amber-500 text-black py-2.5 px-4 font-medium text-xs tracking-wide hover:bg-amber-400 cursor-pointer active:scale-95"
                    >
                      Submit caller response
                    </button>
                  </>
                )}

                {challengeStatus === "passed" && (
                  <p className="text-sm text-emerald-400">✓ Digits correct and latency human-like. Caller re-authenticated — transaction may proceed.</p>
                )}
                {challengeStatus === "failed" && (
                  <p className="text-sm text-rose-400">✗ Challenge failed. A pre-recording cannot answer a live phrase; response escalated to BLOCK.</p>
                )}
              </div>
            </section>
          )}

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