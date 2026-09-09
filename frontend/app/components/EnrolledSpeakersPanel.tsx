"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Users, UserPlus, Loader2, Fingerprint } from "lucide-react";

interface EnrolledSpeakersPanelProps {
  backendUrl: string;
  claimedIdentity: string;
  onSelectIdentity: (userId: string) => void;
  onEvent?: (message: string, severity: "info" | "warning" | "critical") => void;
  disabled?: boolean;
}

/**
 * Lists enrolled voices (GET /api/v1/enroll) and lets the operator switch the
 * claimed identity, or enrol a new one (POST /api/v1/enroll) — the UI side of
 * multi-speaker support: the backend's ASV enrollment was already generic
 * per-user_id, this just makes more than one profile reachable instead of
 * the single hardcoded "narendra_modi" anchor.
 *
 * Real biometrics require VR_ASV_BACKEND=ecapa on the backend (see
 * docs/PROTOCOL.md) — the `asv_backend` badge below shows honestly which
 * one is actually running, since the stub's "enrollment" is not a real
 * voiceprint.
 */
export default function EnrolledSpeakersPanel({
  backendUrl,
  claimedIdentity,
  onSelectIdentity,
  onEvent,
  disabled,
}: EnrolledSpeakersPanelProps) {
  const [enrolled, setEnrolled] = useState<string[]>([]);
  const [asvBackend, setAsvBackend] = useState<string>("");
  const [newLabel, setNewLabel] = useState("");
  const [newFile, setNewFile] = useState<File | null>(null);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/api/v1/enroll`);
      if (!res.ok) return;
      const data = await res.json();
      setEnrolled(Array.isArray(data.enrolled) ? data.enrolled : []);
      setAsvBackend(data.asv_backend || "");
    } catch {
      // Backend offline — the rest of the dashboard already surfaces that.
    }
  }, [backendUrl]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleEnroll = async () => {
    const userId = newLabel.trim().toLowerCase().replace(/\s+/g, "_");
    if (!userId || !newFile) return;
    setIsEnrolling(true);
    try {
      const formData = new FormData();
      formData.append("user_id", userId);
      formData.append("audio", newFile, newFile.name);
      const res = await fetch(`${backendUrl}/api/v1/enroll`, { method: "POST", body: formData });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API error (${res.status}): ${text}`);
      }
      onEvent?.(`Enrolled new voice: "${userId}"`, "info");
      setNewLabel("");
      setNewFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await refresh();
      onSelectIdentity(userId);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      onEvent?.(`Enrollment failed: ${msg}`, "critical");
    } finally {
      setIsEnrolling(false);
    }
  };

  const isRealBiometric = asvBackend === "ecapa_tdnn_voxceleb";

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0c0c] p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Users className="h-4 w-4 text-[#a1a1a1]" />
          <div>
            <span className="font-mono text-xs uppercase tracking-wider text-[#a1a1a1]">
              ENROLLED VOICES
            </span>
            <h3 className="mt-0.5 text-sm font-semibold tracking-[-0.4px] text-white">
              Family / household profiles
            </h3>
          </div>
        </div>
        {asvBackend && (
          <div
            className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-[10.5px] font-mono ${
              isRealBiometric
                ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-400"
                : "border-amber-500/25 bg-amber-500/10 text-amber-400"
            }`}
          >
            <Fingerprint className="h-3 w-3" />
            <span>{isRealBiometric ? "Real ECAPA-TDNN biometric" : "Demo biometric (not real voiceprint)"}</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-1.5">
        {enrolled.length === 0 && (
          <span className="text-xs text-[#666666]">No voices enrolled yet.</span>
        )}
        {enrolled.map((userId) => (
          <button
            key={userId}
            type="button"
            disabled={disabled}
            onClick={() => onSelectIdentity(userId)}
            className={`rounded-full border px-3 py-1.5 font-mono text-[11px] transition disabled:cursor-not-allowed disabled:opacity-50 ${
              userId === claimedIdentity
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                : "border-white/[0.12] bg-[#171717] text-[#a1a1a1] hover:border-white/[0.25] hover:text-white"
            }`}
          >
            {userId.replace(/_/g, " ")}
          </button>
        ))}
      </div>

      <div className="mt-4 flex flex-col sm:flex-row gap-2 border-t border-white/[0.06] pt-4">
        <input
          type="text"
          value={newLabel}
          onChange={(e) => setNewLabel(e.target.value)}
          placeholder="Name (e.g. mom, dad)"
          disabled={isEnrolling}
          className="flex-1 rounded-md border border-white/[0.12] bg-[#171717] px-3 py-2 text-xs text-white placeholder:text-[#666666] focus:outline-none focus:border-emerald-500/40"
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
          onChange={(e) => setNewFile(e.target.files?.[0] || null)}
          disabled={isEnrolling}
          className="flex-1 text-[11px] text-[#a1a1a1] file:mr-2 file:rounded file:border-0 file:bg-[#171717] file:px-2 file:py-1.5 file:text-[11px] file:text-[#a1a1a1] file:cursor-pointer"
        />
        <button
          type="button"
          onClick={handleEnroll}
          disabled={isEnrolling || !newLabel.trim() || !newFile}
          className="flex items-center justify-center gap-1.5 rounded-md border border-white/[0.12] bg-white text-black px-3 py-2 text-xs font-medium hover:bg-neutral-200 transition disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isEnrolling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <UserPlus className="h-3.5 w-3.5" />}
          <span>Enroll</span>
        </button>
      </div>
      <p className="mt-2 text-[10.5px] leading-relaxed text-[#666666]">
        10-20s of clear speech works best. No raw audio is retained — only the derived voiceprint (DPDP Act 2023 §8(7)).
      </p>
    </div>
  );
}
