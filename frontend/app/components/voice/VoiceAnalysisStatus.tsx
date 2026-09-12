"use client";

import React, { useEffect, useState } from "react";

interface VoiceAnalysisStatusProps {
  isListening: boolean;
  audioLevel: number;
  customMessage?: string;
  className?: string;
}

const STATUS_MESSAGES = [
  "Listening…",
  "Analyzing voice…",
  "Inspecting acoustic signature…",
  "Checking speaker consistency…",
  "Evaluating authenticity…",
  "Running forensic analysis…",
];

export default function VoiceAnalysisStatus({
  isListening,
  audioLevel,
  customMessage,
  className = "",
}: VoiceAnalysisStatusProps) {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (!isListening) return;

    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % STATUS_MESSAGES.length);
    }, 2800);

    return () => clearInterval(interval);
  }, [isListening]);

  const activeMessage = customMessage || (!isListening ? "Listening…" : audioLevel > 15 ? STATUS_MESSAGES[messageIndex] : "Listening…");

  return (
    <div className={`flex flex-col items-center text-center select-none ${className}`}>
      <div className="h-9 flex items-center justify-center">
        <span className="text-xl sm:text-2xl font-medium tracking-tight text-white transition-opacity duration-300">
          {activeMessage}
        </span>
      </div>

      <p className="mt-1.5 text-xs sm:text-sm text-slate-400 tracking-wide font-normal">
        {audioLevel > 20 ? "Acoustic signal captured" : "Speak naturally for continuous analysis"}
      </p>
    </div>
  );
}
