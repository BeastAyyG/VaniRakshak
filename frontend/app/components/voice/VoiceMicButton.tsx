"use client";

import React from "react";
import { Mic } from "lucide-react";

interface VoiceMicButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isReady?: boolean;
  size?: "default" | "large";
  className?: string;
}

export default function VoiceMicButton({
  onClick,
  disabled = false,
  isReady = true,
  size = "default",
  className = "",
}: VoiceMicButtonProps) {
  const handleClick = () => {
    if (disabled) return;
    try {
      if (typeof window !== "undefined" && "vibrate" in navigator) {
        navigator.vibrate(25);
      }
    } catch {
      // Haptics unavailable, silently proceed
    }
    onClick();
  };

  const dimClass = size === "large" ? "w-44 h-44 sm:w-48 sm:h-48" : "w-36 h-36 sm:w-40 sm:h-40";

  return (
    <div className={`relative flex items-center justify-center select-none ${className}`}>
      {/* Outer ambient glow ripple layer */}
      <div
        className={`absolute inset-0 rounded-full bg-cyan-500/15 blur-2xl transition-all duration-1000 ${
          isReady ? "animate-pulse-ring" : "opacity-30"
        }`}
      />

      {/* Secondary concentric glass aura ring */}
      <div className="absolute -inset-3.5 sm:-inset-4 rounded-full border border-cyan-400/20 bg-cyan-950/10 backdrop-blur-sm pointer-events-none" />

      {/* Tertiary subtle perimeter boundary */}
      <div className="absolute -inset-7 sm:-inset-8 rounded-full border border-white/[0.05] pointer-events-none" />

      {/* Hero Circular Glass Orb */}
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled}
        aria-label="Start voice analysis"
        className={`relative ${dimClass} rounded-full flex flex-col items-center justify-center cursor-pointer 
          bg-gradient-to-b from-white/[0.12] via-cyan-950/40 to-[#05111d]
          border border-cyan-300/30 shadow-[0_0_50px_rgba(6,182,212,0.25)]
          hover:shadow-[0_0_75px_rgba(6,182,212,0.45)] hover:border-cyan-300/50 hover:scale-105
          active:scale-95 transition-all duration-300 ease-out group focus:outline-none focus:ring-2 focus:ring-cyan-400/50`}
      >
        {/* Specular glass reflection cap */}
        <div className="absolute top-2.5 inset-x-8 h-10 rounded-full bg-gradient-to-b from-white/30 to-transparent blur-[1px] pointer-events-none" />

        {/* Inner glowing pulse ring */}
        <div className="absolute inset-2 rounded-full border border-white/10 bg-radial from-cyan-400/10 via-transparent to-transparent pointer-events-none" />

        {/* Center Mic Icon */}
        <div className="relative flex items-center justify-center rounded-full bg-cyan-500/20 p-5 border border-cyan-400/30 group-hover:bg-cyan-400/30 transition-colors">
          <Mic className="w-8 h-8 sm:w-9 sm:h-9 text-cyan-200 group-hover:text-white transition-colors" />
        </div>
      </button>
    </div>
  );
}
