document.addEventListener("DOMContentLoaded", () => {
  // Modern SVG Icons for crisp, professional rendering (No emojis or font icons)
  const SVG_ICONS = {
    mic: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>`,
    stop: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="3"></rect></svg>`,
    spark: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>`,
    shield: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>`,
    alert: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`,
    critical: `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`,
    spinner: `<div class="loading-radar" style="width: 14px; height: 14px; border-width: 2px; margin: 0; display: inline-block;"></div>`,
    check: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
    cross: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`,
    question: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`
  };

  // Helper: Escape HTML string to prevent injection
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatMMSS(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  // ------------------------------------------------------------------------
  // 1. TAB & MODE NAVIGATION (SYNCHRONIZED)
  // ------------------------------------------------------------------------
  const tabVoice = document.getElementById("tab-voice");
  const tabText = document.getElementById("tab-text");
  const sectionVoice = document.getElementById("section-voice");
  const sectionText = document.getElementById("section-text");

  function switchToVoiceTab() {
    tabVoice.classList.add("active");
    tabText.classList.remove("active");
    sectionVoice.classList.add("active");
    sectionText.classList.remove("active");
  }

  function switchToTextTab() {
    tabText.classList.add("active");
    tabVoice.classList.remove("active");
    sectionText.classList.add("active");
    sectionVoice.classList.remove("active");
  }

  if (tabVoice) tabVoice.addEventListener("click", switchToVoiceTab);
  if (tabText) tabText.addEventListener("click", switchToTextTab);

  // ------------------------------------------------------------------------
  // 2. AUDIO VISUALIZER (CALM IDLE WAVE + LIVE AUDIO FREQUENCY BARS)
  // ------------------------------------------------------------------------
  const visualizerCanvas = document.getElementById("visualizer-canvas");
  let audioContext = null;
  let analyserNode = null;
  let visualizerAnimId = null;

  function drawIdleVisualizer() {
    if (!visualizerCanvas) return;
    const ctx = visualizerCanvas.getContext("2d");
    const w = visualizerCanvas.width;
    const h = visualizerCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    const bars = 48;
    const barWidth = 4;
    const spacing = (w - (bars * barWidth)) / (bars - 1);
    ctx.fillStyle = "#e2e0d8";

    for (let i = 0; i < bars; i++) {
      const norm = i / (bars - 1);
      const wave = Math.sin(norm * Math.PI) * 0.7 + Math.sin(norm * Math.PI * 3) * 0.3;
      const barH = Math.max(3, Math.abs(wave) * (h * 0.55));
      const x = i * (barWidth + spacing);
      const y = (h - barH) / 2;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barH, 2);
      ctx.fill();
    }
  }

  function setupAudioVisualizer(stream) {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 64;
      source.connect(analyserNode);

      const ctx = visualizerCanvas.getContext("2d");
      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      function draw() {
        visualizerAnimId = requestAnimationFrame(draw);
        analyserNode.getByteFrequencyData(dataArray);

        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);

        const barWidth = (visualizerCanvas.width / bufferLength) * 1.4;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = Math.max(3, (dataArray[i] / 255) * visualizerCanvas.height);
          ctx.fillStyle = "#1c1917";
          ctx.beginPath();
          ctx.roundRect(x, visualizerCanvas.height - barHeight, barWidth - 1.5, barHeight, 2);
          ctx.fill();
          x += barWidth;
        }
      }

      draw();
    } catch (e) {
      console.warn("Visualizer init warning:", e);
    }
  }

  function cleanupVisualizer() {
    if (visualizerAnimId) cancelAnimationFrame(visualizerAnimId);
    if (audioContext && audioContext.state !== "closed") {
      audioContext.close().catch(() => {});
    }
    drawIdleVisualizer();
  }

  // Draw initial idle visualizer
  drawIdleVisualizer();

  // ------------------------------------------------------------------------
  // 3. VOICE GUARDIAN STATE & UI ELEMENTS
  // ------------------------------------------------------------------------
  let currentAudioBlob = null;
  let currentAudioFileName = "";
  let mediaRecorder = null;
  let audioChunks = [];
  let recordTimer = null;
  let elapsedSeconds = 0;
  const MAX_RECORD_SECONDS = 60;

  const micBtn = document.getElementById("mic-btn");
  const micIcon = document.getElementById("mic-icon");
  const micLabel = document.getElementById("mic-label");
  const micHint = document.getElementById("mic-hint");
  const timerDisplay = document.getElementById("timer-display");
  const recordProgressFill = document.getElementById("record-progress-fill");
  const autoDetectCheckbox = document.getElementById("auto-detect-checkbox");

  const dropzone = document.getElementById("dropzone");
  const audioFileInput = document.getElementById("audio-file-input");
  const selectedFileInfo = document.getElementById("selected-file-info");
  const fileNameText = document.getElementById("file-name");
  const clearFileBtn = document.getElementById("clear-file-btn");

  const audioPreviewWrapper = document.getElementById("audio-preview-wrapper");
  const audioPlayer = document.getElementById("audio-player");

  const analyzeVoiceBtn = document.getElementById("analyze-voice-btn");
  const voiceBtnText = document.getElementById("voice-btn-text");
  const voiceBtnIcon = document.getElementById("voice-btn-icon");

  const voiceLoadingBox = document.getElementById("voice-loading-box");
  const voiceLoadingStatus = document.getElementById("voice-loading-status");
  const voiceLoadingDetail = document.getElementById("voice-loading-detail");

  const voiceResultsBox = document.getElementById("voice-results-box");
  const threatHero = document.getElementById("threat-hero");
  const threatBadge = document.getElementById("threat-badge");
  const threatIcon = document.getElementById("threat-icon");
  const threatText = document.getElementById("threat-text");
  const scamTypeTag = document.getElementById("scam-type-tag");
  const threatScore = document.getElementById("threat-score");
  const scoreMeterFill = document.getElementById("score-meter-fill");

  const emergencyAdviceBox = document.getElementById("emergency-advice-box");
  const adviceText = document.getElementById("advice-text");

  const langPill = document.getElementById("lang-pill");
  const latencyPill = document.getElementById("latency-pill");
  const durationPill = document.getElementById("duration-pill");
  const transcribedTextBox = document.getElementById("transcribed-text-box");

  const segmentsSection = document.getElementById("segments-section");
  const segmentsList = document.getElementById("segments-list");

  const redFlagsSection = document.getElementById("red-flags-section");
  const redFlagsList = document.getElementById("red-flags-list");

  const voiceFactSection = document.getElementById("voice-fact-section");
  const voiceFactVerdict = document.getElementById("voice-fact-verdict");
  const voiceFactConfidence = document.getElementById("voice-fact-confidence");
  const voiceFactSummary = document.getElementById("voice-fact-summary");
  const voiceFactReasoning = document.getElementById("voice-fact-reasoning");
  const voiceSourcesContainer = document.getElementById("voice-sources-container");

  // ------------------------------------------------------------------------
  // 4. SYNCHRONIZED FILE UPLOAD & MIC RECORDING
  // ------------------------------------------------------------------------
  // Reset all audio state
  function resetAudioState() {
    currentAudioBlob = null;
    currentAudioFileName = "";
    if (audioFileInput) audioFileInput.value = "";
    if (selectedFileInfo) selectedFileInfo.style.display = "none";
    if (audioPreviewWrapper) audioPreviewWrapper.style.display = "none";
    if (audioPlayer) {
      audioPlayer.pause();
      audioPlayer.src = "";
    }
    if (timerDisplay) timerDisplay.textContent = "00:00 / 01:00";
    if (recordProgressFill) recordProgressFill.style.width = "0%";
    if (analyzeVoiceBtn) analyzeVoiceBtn.disabled = true;
    cleanupVisualizer();
  }

  // Handle newly selected audio file (via drop or file picker)
  function handleAudioFile(file) {
    if (!file) return;

    // If mic is actively recording, stop it cleanly
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording(false); // don't auto-analyze recorded fragment
    }

    currentAudioBlob = file;
    currentAudioFileName = file.name;

    selectedFileInfo.style.display = "flex";
    fileNameText.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;

    audioPlayer.src = URL.createObjectURL(file);
    audioPreviewWrapper.style.display = "block";

    analyzeVoiceBtn.disabled = false;
    voiceBtnText.textContent = "Transcribe & Analyze Audio";
  }

  // Dropzone click & drag events
  if (dropzone && audioFileInput) {
    dropzone.addEventListener("click", () => audioFileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleAudioFile(e.dataTransfer.files[0]);
      }
    });

    audioFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleAudioFile(e.target.files[0]);
      }
    });
  }

  // Clear file button
  if (clearFileBtn) {
    clearFileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      resetAudioState();
    });
  }

  // Microphone toggle button
  if (micBtn) {
    micBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording(true);
      } else {
        startRecording();
      }
    });
  }

  async function startRecording() {
    try {
      // Clear any uploaded file first to maintain state sync
      resetAudioState();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      elapsedSeconds = 0;

      setupAudioVisualizer(stream);

      const options = { mimeType: "audio/webm" };
      if (!MediaRecorder.isTypeSupported("audio/webm")) {
        delete options.mimeType;
      }
      mediaRecorder = new MediaRecorder(stream, options);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const mime = mediaRecorder.mimeType || "audio/webm";
        const extension = mime.includes("webm") ? "webm" : "wav";
        currentAudioBlob = new Blob(audioChunks, { type: mime });
        currentAudioFileName = `live_speech_${Date.now()}.${extension}`;

        audioPlayer.src = URL.createObjectURL(currentAudioBlob);
        audioPreviewWrapper.style.display = "block";

        selectedFileInfo.style.display = "flex";
        fileNameText.textContent = `Live Voice Recording (${formatMMSS(elapsedSeconds)} / 60s max)`;

        analyzeVoiceBtn.disabled = false;
        voiceBtnText.textContent = "Transcribe & Analyze Audio";

        stream.getTracks().forEach(t => t.stop());
        cleanupVisualizer();

        // Auto-detect immediately if enabled
        if (autoDetectCheckbox && autoDetectCheckbox.checked && elapsedSeconds > 0) {
          setTimeout(() => {
            handleVoiceAnalysis();
          }, 250);
        }
      };

      mediaRecorder.start();
      micBtn.classList.add("recording");
      micIcon.innerHTML = SVG_ICONS.stop;
      micLabel.textContent = "Recording Speech...";
      micHint.textContent = "Click stop anytime or auto-stops at 60s";

      timerDisplay.textContent = "00:00 / 01:00";
      if (recordProgressFill) recordProgressFill.style.width = "0%";

      recordTimer = setInterval(() => {
        elapsedSeconds += 1;
        timerDisplay.textContent = `${formatMMSS(elapsedSeconds)} / 01:00`;

        if (recordProgressFill) {
          const pct = Math.min(100, (elapsedSeconds / MAX_RECORD_SECONDS) * 100);
          recordProgressFill.style.width = `${pct}%`;
        }

        if (elapsedSeconds >= MAX_RECORD_SECONDS) {
          stopRecording(true);
        }
      }, 1000);

    } catch (err) {
      console.error("Mic access denied:", err);
      alert("Microphone access failed: " + err.message + "\nPlease grant microphone permissions or drop an audio file.");
    }
  }

  function stopRecording(shouldProcess = true) {
    if (recordTimer) {
      clearInterval(recordTimer);
      recordTimer = null;
    }
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    micBtn.classList.remove("recording");
    micIcon.innerHTML = SVG_ICONS.mic;
    micLabel.textContent = "Recording Ready";
    micHint.textContent = "Audio captured — click analyze below to evaluate threat";
  }

  // ------------------------------------------------------------------------
  // 5. AUDIO SUBMISSION & ANALYSIS PIPELINE
  // ------------------------------------------------------------------------
  if (analyzeVoiceBtn) {
    analyzeVoiceBtn.addEventListener("click", handleVoiceAnalysis);
  }

  function cycleVoiceLoadingMessages() {
    const msgs = [
      { status: "Transcribing speech audio...", detail: "Recognizing spoken Hindi, Hinglish, and English dialects" },
      { status: "Analyzing threat & scam markers...", detail: "Evaluating digital arrest coercion, impersonation, and pressure tactics" },
      { status: "Grounding evidence with live web verification...", detail: "Cross-referencing claims against authoritative advisories and sources" },
      { status: "Synthesizing defense intelligence...", detail: "Finalizing risk score, verdict, and immediate directive" }
    ];
    let idx = 0;
    return setInterval(() => {
      idx = (idx + 1) % msgs.length;
      if (voiceLoadingStatus) voiceLoadingStatus.textContent = msgs[idx].status;
      if (voiceLoadingDetail) voiceLoadingDetail.textContent = msgs[idx].detail;
    }, 2400);
  }

  function setVoiceLoading(isLoading) {
    if (isLoading) {
      analyzeVoiceBtn.disabled = true;
      voiceBtnText.textContent = "Analyzing Audio & Grounding Evidence...";
      voiceBtnIcon.innerHTML = SVG_ICONS.spinner;
      voiceLoadingBox.style.display = "flex";
      voiceResultsBox.style.display = "none";
      voiceLoadingBox.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      analyzeVoiceBtn.disabled = false;
      voiceBtnText.textContent = "Transcribe & Analyze Audio";
      voiceBtnIcon.innerHTML = SVG_ICONS.spark;
      voiceLoadingBox.style.display = "none";
    }
  }

  async function handleVoiceAnalysis() {
    if (!currentAudioBlob) {
      alert("Please record or select an audio file first.");
      return;
    }

    setVoiceLoading(true);
    const msgTimer = cycleVoiceLoadingMessages();

    try {
      const formData = new FormData();
      formData.append("file", currentAudioBlob, currentAudioFileName);
      formData.append("verify_fact", "true");

      const response = await fetch("/api/audio/verify-chunk", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Server error occurred." }));
        throw new Error(errData.detail || `HTTP Error ${response.status}`);
      }

      const data = await response.json();
      renderVoiceResults(data);
    } catch (err) {
      console.error(err);
      alert("Voice verification error: " + err.message);
    } finally {
      clearInterval(msgTimer);
      setVoiceLoading(false);
    }
  }

  function renderVoiceResults(data) {
    const t = data.transcription || {};
    const threat = data.threat || {};
    const fact = data.fact_check || null;

    // 1. Transcription details
    if (langPill) langPill.textContent = t.detected_language ? `Language: ${t.detected_language}` : "English / Hindi";
    if (latencyPill) latencyPill.textContent = `Response: ${data.pipeline_time_sec || 1.1}s`;
    if (durationPill) durationPill.textContent = `Duration: ${t.duration ? t.duration.toFixed(1) + "s" : "Captured"}`;

    if (t.transcript && t.transcript.trim().length > 0) {
      const hindiText = t.hindi_text || t.transcript;
      const hinglishText = t.hinglish_text || "";
      if (hinglishText && hinglishText !== hindiText) {
        transcribedTextBox.innerHTML = `
          <div style="margin-bottom: 8px;"><strong>Spoken Transcript:</strong> "${escapeHtml(hindiText)}"</div>
          <div style="color: #059669; border-top: 1px dashed rgba(13,12,10,0.12); padding-top: 8px;"><strong>Phonetic Dialect:</strong> "${escapeHtml(hinglishText)}"</div>
        `;
      } else {
        transcribedTextBox.textContent = `"${hindiText}"`;
      }
    } else {
      transcribedTextBox.textContent = "No audible voice or intelligible speech detected.";
    }

    // 2. Timeline Segments
    if (segmentsSection && segmentsList) {
      if (t.segments && t.segments.length > 0) {
        segmentsSection.style.display = "block";
        segmentsList.innerHTML = "";
        t.segments.forEach(seg => {
          const item = document.createElement("div");
          item.className = "segment-item";
          item.innerHTML = `
            <span class="segment-time">[${seg.start.toFixed(1)}s ➔ ${seg.end.toFixed(1)}s]</span>
            <span class="segment-text">${escapeHtml(seg.text)}</span>
          `;
          segmentsList.appendChild(item);
        });
      } else {
        segmentsSection.style.display = "none";
      }
    }

    // 3. Threat Level Hero Banner
    const level = (threat.threat_level || "SAFE").toUpperCase();
    const score = Math.min(100, Math.max(0, threat.risk_score || 0));

    threatHero.className = "threat-hero";
    if (level.includes("CRITICAL") || score >= 75) {
      threatHero.classList.add("critical");
      threatIcon.innerHTML = SVG_ICONS.critical;
      threatText.textContent = "CRITICAL SCAM DETECTED";
      scoreMeterFill.style.backgroundColor = "#e11d48";
    } else if (level.includes("SUSPICIOUS") || score >= 40) {
      threatHero.classList.add("suspicious");
      threatIcon.innerHTML = SVG_ICONS.alert;
      threatText.textContent = "SUSPICIOUS COERCION / CLAIM";
      scoreMeterFill.style.backgroundColor = "#d97706";
    } else {
      threatHero.classList.add("safe");
      threatIcon.innerHTML = SVG_ICONS.shield;
      threatText.textContent = "SAFE / NORMAL CONVERSATION";
      scoreMeterFill.style.backgroundColor = "#059669";
    }

    scamTypeTag.textContent = threat.scam_type || "Normal Conversation";
    threatScore.textContent = `${score} / 100`;
    scoreMeterFill.style.width = `${score}%`;

    // 4. Emergency Action Advisory
    if (threat.action_advice) {
      adviceText.textContent = threat.action_advice;
      emergencyAdviceBox.style.display = "block";
    } else {
      emergencyAdviceBox.style.display = "none";
    }

    // 5. Red Flags
    redFlagsList.innerHTML = "";
    if (threat.red_flags && threat.red_flags.length > 0) {
      redFlagsSection.style.display = "block";
      threat.red_flags.forEach(flag => {
        const item = document.createElement("div");
        item.className = "red-flag-item";
        item.innerHTML = `<span>⚠️</span><span>${escapeHtml(flag)}</span>`;
        redFlagsList.appendChild(item);
      });
    } else {
      redFlagsSection.style.display = "none";
    }

    // 6. Grounded Web Evidence
    if (fact && fact.verdict) {
      voiceFactSection.style.display = "block";
      const v = fact.verdict.toUpperCase();
      voiceFactVerdict.textContent = v;
      voiceFactConfidence.textContent = `Confidence: ${fact.confidence || 0}%`;

      if (v.includes("SUPPORTED") || v.includes("TRUE")) {
        voiceFactVerdict.style.backgroundColor = "var(--success-bg)";
        voiceFactVerdict.style.color = "var(--success-ink)";
      } else if (v.includes("CONTRADICTED") || v.includes("FRAUD") || v.includes("FALSE")) {
        voiceFactVerdict.style.backgroundColor = "var(--error-bg)";
        voiceFactVerdict.style.color = "var(--error-ink)";
      } else {
        voiceFactVerdict.style.backgroundColor = "var(--warning-bg)";
        voiceFactVerdict.style.color = "var(--warning-ink)";
      }

      voiceFactSummary.textContent = fact.summary || "";
      voiceFactReasoning.textContent = fact.reasoning || "";
    } else {
      voiceFactSection.style.display = "none";
    }

    // 7. Authoritative Web Sources
    voiceSourcesContainer.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach(src => {
        const card = document.createElement("div");
        card.className = "source-item";
        card.innerHTML = `
          <div class="source-header">
            <div class="source-title">${escapeHtml(src.title || "Web Advisory")}</div>
            <div class="source-url">
              <a href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(src.url)} ↗</a>
            </div>
          </div>
          <div class="source-snippet">${escapeHtml(src.snippet || "Grounded web source extracted.")}</div>
        `;
        voiceSourcesContainer.appendChild(card);
      });
    } else {
      voiceSourcesContainer.innerHTML = `<div style="color: var(--muted); font-size: 0.85rem;">Autonomous evaluation concluded. No additional web scrapes needed.</div>`;
    }

    voiceResultsBox.style.display = "block";
    voiceResultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------------
  // 6. TAB 2: DIRECT TEXT FACT CHECKER
  // ------------------------------------------------------------------------
  const claimInput = document.getElementById("claim-input");
  const urlInput = document.getElementById("url-input");
  const verifyTextBtn = document.getElementById("verify-text-btn");
  const textBtnText = document.getElementById("text-btn-text");
  const textBtnIcon = document.getElementById("text-btn-icon");

  const textLoadingBox = document.getElementById("text-loading-box");
  const textLoadingStatus = document.getElementById("text-loading-status");
  const textLoadingDetail = document.getElementById("text-loading-detail");

  const textResultsBox = document.getElementById("text-results-box");
  const textVerdictHero = document.getElementById("text-verdict-hero");
  const textVerdictBadge = document.getElementById("text-verdict-badge");
  const textVerdictIcon = document.getElementById("text-verdict-icon");
  const textVerdictText = document.getElementById("text-verdict-text");
  const textConfidenceVal = document.getElementById("text-confidence-val");
  const textConfidenceFill = document.getElementById("text-confidence-fill");

  const textSummaryContent = document.getElementById("text-summary-content");
  const textReasoningContent = document.getElementById("text-reasoning-content");
  const textFactsContainer = document.getElementById("text-facts-container");
  const textSourcesContainer = document.getElementById("text-sources-container");

  // Preset quick-test pills
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      claimInput.value = btn.getAttribute("data-claim") || "";
      urlInput.value = btn.getAttribute("data-urls") || "";
      claimInput.focus();
    });
  });

  // Ctrl+Enter / Cmd+Enter shortcut
  if (claimInput) {
    claimInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleTextVerification();
      }
    });
  }

  if (verifyTextBtn) {
    verifyTextBtn.addEventListener("click", handleTextVerification);
  }

  function setTextLoading(isLoading) {
    if (isLoading) {
      verifyTextBtn.disabled = true;
      textBtnText.textContent = "Verifying with Evidence Engine...";
      textBtnIcon.innerHTML = SVG_ICONS.spinner;
      textLoadingBox.style.display = "flex";
      textResultsBox.style.display = "none";
      textLoadingBox.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      verifyTextBtn.disabled = false;
      textBtnText.textContent = "Detect & Verify Fact";
      textBtnIcon.innerHTML = SVG_ICONS.spark;
      textLoadingBox.style.display = "none";
    }
  }

  function cycleTextLoadingMessages() {
    const msgs = [
      { status: "Searching live web...", detail: "Navigating authoritative domains and fact repositories" },
      { status: "Scraping page evidence...", detail: "Extracting verified textual content and timestamps" },
      { status: "Analyzing claims...", detail: "Evaluating claim against scraped evidence with strict grounding" }
    ];
    let idx = 0;
    return setInterval(() => {
      idx = (idx + 1) % msgs.length;
      if (textLoadingStatus) textLoadingStatus.textContent = msgs[idx].status;
      if (textLoadingDetail) textLoadingDetail.textContent = msgs[idx].detail;
    }, 2400);
  }

  async function handleTextVerification() {
    const claim = claimInput.value.trim();
    if (!claim) {
      alert("Please enter a claim or statement to fact-check.");
      claimInput.focus();
      return;
    }

    const rawUrls = urlInput ? urlInput.value.trim() : "";
    let urls = null;
    if (rawUrls) {
      urls = rawUrls.split(",").map(u => u.trim()).filter(u => u.length > 0);
    }

    setTextLoading(true);
    const timer = cycleTextLoadingMessages();

    try {
      const response = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim, urls })
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Network error occurred." }));
        throw new Error(err.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      renderTextResults(data);
    } catch (err) {
      console.error(err);
      alert("Fact check error: " + err.message);
    } finally {
      clearInterval(timer);
      setTextLoading(false);
    }
  }

  function renderTextResults(data) {
    const verdict = (data.verdict || "UNVERIFIABLE").toUpperCase();
    const confidence = Math.round(data.confidence || 0);

    textVerdictHero.className = "verdict-hero";
    textVerdictBadge.className = "verdict-badge";

    let iconSvg = SVG_ICONS.question;
    let statusClass = "unverifiable";

    if (verdict.includes("SUPPORTED") || verdict.includes("TRUE")) {
      iconSvg = SVG_ICONS.check;
      statusClass = "supported";
    } else if (verdict.includes("CONTRADICTED") || verdict.includes("FALSE") || verdict.includes("FRAUD")) {
      iconSvg = SVG_ICONS.cross;
      statusClass = "contradicted";
    }

    textVerdictHero.classList.add(statusClass);
    textVerdictBadge.classList.add(statusClass);
    textVerdictIcon.innerHTML = iconSvg;
    textVerdictText.textContent = verdict;

    textConfidenceVal.textContent = `${confidence}%`;
    textConfidenceFill.style.width = `${confidence}%`;
    textConfidenceFill.style.backgroundColor = 
      statusClass === "supported" ? "#059669" :
      statusClass === "contradicted" ? "#e11d48" : "#d97706";

    textSummaryContent.textContent = data.summary || "No summary provided.";
    textReasoningContent.textContent = data.reasoning || "Autonomous evidence synthesis complete.";

    // Key facts
    textFactsContainer.innerHTML = "";
    if (data.key_facts && data.key_facts.length > 0) {
      document.getElementById("text-key-facts-section").style.display = "block";
      data.key_facts.forEach(f => {
        const item = document.createElement("div");
        item.className = "fact-item";
        item.innerHTML = `
          <div class="fact-text"><strong>${escapeHtml(f.fact)}</strong></div>
          <div class="fact-source" style="margin-top: 4px; font-size: 0.75rem; color: var(--muted);">Source: ${escapeHtml(f.source || "Grounded Web Evidence")}</div>
        `;
        textFactsContainer.appendChild(item);
      });
    } else {
      document.getElementById("text-key-facts-section").style.display = "none";
    }

    // Sources list
    textSourcesContainer.innerHTML = "";
    if (data.sources && data.sources.length > 0) {
      data.sources.forEach(src => {
        const card = document.createElement("div");
        card.className = "source-item";
        card.innerHTML = `
          <div class="source-title">${escapeHtml(src.title || "Web Evidence Source")}</div>
          <div class="source-url"><a href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(src.url)} ↗</a></div>
          <div class="source-snippet">${escapeHtml(src.snippet || "")}</div>
        `;
        textSourcesContainer.appendChild(card);
      });
    } else {
      textSourcesContainer.innerHTML = `<div style="color: var(--muted); font-size: 0.85rem;">Autonomous validation complete.</div>`;
    }

    textResultsBox.style.display = "block";
    textResultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------------
  // 7. LIVE STREAM CALL MONITOR (WEB SOCKET INTEGRATION)
  // ------------------------------------------------------------------------
  const liveStreamBtn = document.getElementById("simulate-stream-btn");
  const liveStreamPanel = document.getElementById("live-stream-panel");
  const streamStatusPill = document.getElementById("stream-status-pill");
  const streamTranscriptFeed = document.getElementById("stream-transcript-feed");
  const streamThreatScore = document.getElementById("stream-threat-score");
  const streamThreatBar = document.getElementById("stream-threat-bar");
  const streamEmergencyAlert = document.getElementById("stream-emergency-alert");

  let streamSocket = null;
  let simulatedStreamInterval = null;

  function connectCallStreamWebSocket(callId = "call_demo_" + Date.now()) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live-call-stream?call_id=${callId}`;

    if (streamStatusPill) {
      streamStatusPill.textContent = "Connecting to Live Stream...";
      streamStatusPill.className = "stream-pill connecting";
    }

    streamSocket = new WebSocket(wsUrl);

    streamSocket.onopen = () => {
      if (streamStatusPill) {
        streamStatusPill.textContent = "● Live Call Streaming Active";
        streamStatusPill.className = "stream-pill active";
      }
    };

    streamSocket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleStreamUpdate(msg);
      } catch (e) {
        console.warn("WebSocket parse error:", e);
      }
    };

    streamSocket.onclose = () => {
      if (streamStatusPill) {
        streamStatusPill.textContent = "Stream Disconnected (Call Ended)";
        streamStatusPill.className = "stream-pill idle";
      }
    };

    streamSocket.onerror = (err) => {
      console.warn("WebSocket stream error, fallback active:", err);
    };
  }

  function handleStreamUpdate(data) {
    if (data.full_transcript && streamTranscriptFeed) {
      streamTranscriptFeed.textContent = data.full_transcript;
      streamTranscriptFeed.scrollTop = streamTranscriptFeed.scrollHeight;
    }

    if (data.threat_score !== undefined && streamThreatScore && streamThreatBar) {
      streamThreatScore.textContent = `${data.threat_score} / 100`;
      streamThreatBar.style.width = `${data.threat_score}%`;
      streamThreatBar.style.backgroundColor = 
        data.threat_score >= 75 ? "#e11d48" : 
        data.threat_score >= 40 ? "#d97706" : "#059669";
    }

    if (data.action_advice && streamEmergencyAlert) {
      if (data.threat_score >= 70) {
        streamEmergencyAlert.style.display = "block";
        streamEmergencyAlert.textContent = "🚨 CRITICAL DIRECTIVE: " + data.action_advice;
      } else {
        streamEmergencyAlert.style.display = "none";
      }
    }
  }

  // Simulated Voice Call Stream Button for instant UI testing
  if (liveStreamBtn) {
    liveStreamBtn.addEventListener("click", () => {
      if (simulatedStreamInterval) {
        clearInterval(simulatedStreamInterval);
        simulatedStreamInterval = null;
        if (streamSocket) streamSocket.close();
        liveStreamBtn.textContent = "Test Live Call Stream";
        if (streamStatusPill) {
          streamStatusPill.textContent = "Stream Idle";
          streamStatusPill.className = "stream-pill idle";
        }
        return;
      }

      liveStreamBtn.textContent = "Stop Stream Simulation";
      connectCallStreamWebSocket();

      // Simulate a real-world digital arrest scam call progression
      const phrases = [
        "Aapka naam Saumya hai? Hum Mumbai Cyber Crime Cell se bol rahe hain.",
        "Aapke Aadhaar card se ek FedEx parcel bheja gaya hai jisme illegal narcotics aur fake passports mile hain.",
        "Supreme Court aur CBI ne aapke khilaf immediate Digital Arrest warrant issue kiya hai.",
        "Aap kisi ko call nahi kar sakte, camera on rakhiye nahi to police aapke ghar bhej rahe hain.",
        "Verification clearance ke liye aapko Reserve Bank ke security escrow account mein 50,000 transfer karne honge."
      ];

      let phraseIndex = 0;
      let accumulated = "";

      simulatedStreamInterval = setInterval(() => {
        if (phraseIndex < phrases.length) {
          accumulated += (accumulated ? " " : "") + phrases[phraseIndex];
          phraseIndex++;

          const score = phraseIndex === 1 ? 15 : phraseIndex === 2 ? 45 : phraseIndex === 3 ? 82 : 95;
          const advice = score >= 75 
            ? "HANG UP IMMEDIATELY. Law enforcement never arrests via video call or demands money transfers. Call national helpline 1930."
            : "Continue monitoring. Urgency detected.";

          handleStreamUpdate({
            full_transcript: accumulated,
            threat_score: score,
            action_advice: advice
          });

          // Also send text frame to WebSocket if open
          if (streamSocket && streamSocket.readyState === WebSocket.OPEN) {
            streamSocket.send(JSON.stringify({
              event: "text_chunk",
              text: phrases[phraseIndex - 1]
            }));
          }
        } else {
          clearInterval(simulatedStreamInterval);
          simulatedStreamInterval = null;
          liveStreamBtn.textContent = "Test Live Call Stream";
        }
      }, 2500);
    });
  }

});
