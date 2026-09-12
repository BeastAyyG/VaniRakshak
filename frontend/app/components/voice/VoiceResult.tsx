"use client";

import React from "react";
import VerifiedResult from "./VerifiedResult";
import SuspiciousResult from "./SuspiciousResult";
import CriticalResult from "./CriticalResult";

interface VoiceResultProps {
  tier: string; // "ALLOW" | "CHALLENGE" | "BLOCK"
  riskScore: number;
  verdict?: string;
  speakerSimilarity?: number | null;
  snrDb?: number | null;
  challengePhrase?: string;
  expectedDigits?: string[];
  challengeStatus: "none" | "awaiting" | "passed" | "failed";
  onSubmitChallenge: (responseText: string) => Promise<void>;
  onReset: () => void;
  onViewDeepDetails?: () => void;
}

export default function VoiceResult({
  tier,
  riskScore,
  verdict,
  speakerSimilarity,
  snrDb,
  challengePhrase,
  expectedDigits,
  challengeStatus,
  onSubmitChallenge,
  onReset,
  onViewDeepDetails,
}: VoiceResultProps) {
  if (tier === "BLOCK" || riskScore >= 70) {
    return (
      <CriticalResult
        riskScore={riskScore}
        verdict={verdict}
        onReset={onReset}
        onViewDeepDetails={onViewDeepDetails}
      />
    );
  }

  if (tier === "CHALLENGE" || (riskScore >= 35 && riskScore < 70)) {
    return (
      <SuspiciousResult
        riskScore={riskScore}
        challengePhrase={challengePhrase}
        expectedDigits={expectedDigits}
        challengeStatus={challengeStatus}
        onSubmitChallenge={onSubmitChallenge}
        onReset={onReset}
      />
    );
  }

  // Default ALLOW / Low risk
  return (
    <VerifiedResult
      riskScore={riskScore}
      speakerSimilarity={speakerSimilarity}
      snrDb={snrDb}
      onReset={onReset}
      onViewDeepDetails={onViewDeepDetails}
    />
  );
}
