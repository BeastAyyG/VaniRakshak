"use client";

import { useEffect, useRef } from "react";

interface VoiceVisualizerProps {
  audioLevel: number; // 0 to 100 representing live microphone volume
  isListening: boolean;
  state?: "listening" | "processing" | "verified" | "suspicious" | "critical";
  className?: string;
}

export default function VoiceVisualizer({
  audioLevel,
  isListening,
  state = "listening",
  className = "",
}: VoiceVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioLevelRef = useRef<number>(audioLevel);
  const smoothedLevelRef = useRef<number>(audioLevel);
  const rotationRef = useRef<number>(0);

  useEffect(() => {
    audioLevelRef.current = audioLevel;
  }, [audioLevel]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;

      // Smooth the level transition
      smoothedLevelRef.current += (audioLevelRef.current - smoothedLevelRef.current) * 0.2;
      const normLevel = Math.min(1.0, smoothedLevelRef.current / 85);

      rotationRef.current += 0.015 + normLevel * 0.03;
      const rot = rotationRef.current;

      ctx.clearRect(0, 0, width, height);

      // Determine palette based on analysis state
      let coreColor = "rgba(6, 182, 212, "; // Cyan default
      let accentColor = "rgba(59, 130, 246, "; // Blue
      let ringColor = "rgba(14, 165, 233, ";

      if (state === "critical") {
        coreColor = "rgba(244, 63, 94, "; // Rose/red
        accentColor = "rgba(225, 29, 72, ";
        ringColor = "rgba(251, 113, 133, ";
      } else if (state === "suspicious") {
        coreColor = "rgba(245, 158, 11, "; // Amber
        accentColor = "rgba(217, 119, 6, ";
        ringColor = "rgba(252, 211, 77, ";
      } else if (state === "verified") {
        coreColor = "rgba(16, 185, 129, "; // Emerald
        accentColor = "rgba(5, 150, 105, ";
        ringColor = "rgba(110, 231, 183, ";
      }

      // Base radius scaled to canvas size
      const minDim = Math.min(width, height);
      const baseRadius = minDim * 0.22;
      const dynamicRadius = baseRadius + normLevel * minDim * 0.15;

      // 1. Outermost Ambient Atmospheric Glow
      const ambientGrad = ctx.createRadialGradient(
        centerX,
        centerY,
        baseRadius * 0.5,
        centerX,
        centerY,
        dynamicRadius * 2.1
      );
      ambientGrad.addColorStop(0, `${coreColor}${0.35 + normLevel * 0.35})`);
      ambientGrad.addColorStop(0.5, `${accentColor}${0.12 + normLevel * 0.2})`);
      ambientGrad.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.fillStyle = ambientGrad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, dynamicRadius * 2.1, 0, Math.PI * 2);
      ctx.fill();

      // 2. Translucent Concentric Rings (Waveform Echoes)
      const ringCount = 4;
      for (let i = 1; i <= ringCount; i++) {
        const ringProgress = i / ringCount;
        const ringRad = baseRadius + (dynamicRadius - baseRadius + 30 * i) * (0.8 + 0.3 * Math.sin(rot * 0.8 + i));
        const alpha = Math.max(0.04, (1 - ringProgress) * 0.35 + normLevel * 0.25);

        ctx.strokeStyle = `${ringColor}${alpha})`;
        ctx.lineWidth = 1.5 + normLevel * 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, ringRad, 0, Math.PI * 2);
        ctx.stroke();
      }

      // 3. Deforming Radial Waveform (Speech frequency harmonics)
      const points = 64;
      ctx.beginPath();
      for (let i = 0; i <= points; i++) {
        const angle = (i / points) * Math.PI * 2;
        // Modulate with multi-frequency harmonic sines
        const wave =
          Math.sin(angle * 4 + rot * 2) * 6 * normLevel +
          Math.sin(angle * 7 - rot * 3) * 4 * normLevel +
          Math.cos(angle * 2 + rot) * 8 * normLevel;

        const r = dynamicRadius + wave;
        const x = centerX + Math.cos(angle) * r;
        const y = centerY + Math.sin(angle) * r;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();
      ctx.strokeStyle = `${coreColor}${0.8 + normLevel * 0.2})`;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // 4. Inner Luminous Core Orb
      const innerGrad = ctx.createRadialGradient(
        centerX,
        centerY - baseRadius * 0.2,
        baseRadius * 0.1,
        centerX,
        centerY,
        baseRadius * 0.95
      );
      innerGrad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
      innerGrad.addColorStop(0.35, `${coreColor}0.85)`);
      innerGrad.addColorStop(0.85, `${accentColor}0.5)`);
      innerGrad.addColorStop(1, "rgba(5, 12, 20, 0.9)");

      ctx.fillStyle = innerGrad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius * 0.95, 0, Math.PI * 2);
      ctx.fill();

      // 5. Specular Glass Highlight Crescent
      ctx.save();
      ctx.beginPath();
      ctx.arc(centerX, centerY - baseRadius * 0.35, baseRadius * 0.55, 0, Math.PI * 2);
      const specGrad = ctx.createLinearGradient(
        centerX,
        centerY - baseRadius * 0.9,
        centerX,
        centerY + baseRadius * 0.2
      );
      specGrad.addColorStop(0, "rgba(255, 255, 255, 0.45)");
      specGrad.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = specGrad;
      ctx.fill();
      ctx.restore();

      animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isListening, state]);

  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      <canvas
        ref={canvasRef}
        width={340}
        height={340}
        className="w-full max-w-[340px] aspect-square select-none pointer-events-none"
      />
    </div>
  );
}
