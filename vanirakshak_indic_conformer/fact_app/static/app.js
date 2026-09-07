// ==========================================================================
// VANIRAKSHAK — VOICE SCAM SHIELD & FACT DETECTOR JAVASCRIPT CLIENT
// Features:
// 1. Audio Chunk recording via Web Audio API & MediaRecorder.
// 2. Live canvas waveform visualizer.
// 3. Dynamic audio preset loading from /api/audio/samples.
// 4. Drag & drop audio file upload.
// 5. Dual tabs: Voice Guardian & Text Fact Checker.
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  // ------------------------------------------------------------------------
  // TAB NAVIGATION
  // ------------------------------------------------------------------------
  const tabVoice = document.getElementById("tab-voice");
  const tabText = document.getElementById("tab-text");
  const sectionVoice = document.getElementById("section-voice");
  const sectionText = document.getElementById("section-text");

  tabVoice.addEventListener("click", () => {
    tabVoice.classList.add("active");
    tabText.classList.remove("active");
    sectionVoice.classList.add("active");
    sectionText.classList.remove("active");
  });

  tabText.addEventListener("click", () => {
    tabText.classList.add("active");
    tabVoice.classList.remove("active");
    sectionText.classList.add("active");
    sectionVoice.classList.remove("active");
  });

  const heroCtaVoice = document.getElementById("hero-cta-voice");
  const heroCtaText = document.getElementById("hero-cta-text");
  if (heroCtaVoice) heroCtaVoice.addEventListener("click", () => tabVoice.click());
  if (heroCtaText) heroCtaText.addEventListener("click", () => tabText.click());


  // ------------------------------------------------------------------------
  // VOICE STATE & ELEMENTS
  // ------------------------------------------------------------------------
  const samplesContainer = document.getElementById("samples-container");
  const micBtn = document.getElementById("mic-btn");
  const micIcon = document.getElementById("mic-icon");
  const micLabel = document.getElementById("mic-label");
  const micHint = document.getElementById("mic-hint");
  const visualizerCanvas = document.getElementById("visualizer-canvas");
  const timerDisplay = document.getElementById("timer-display");

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

  const redFlagsSection = document.getElementById("red-flags-section");
  const redFlagsList = document.getElementById("red-flags-list");

  const voiceFactSection = document.getElementById("voice-fact-section");
  const voiceFactVerdict = document.getElementById("voice-fact-verdict");
  const voiceFactConfidence = document.getElementById("voice-fact-confidence");
  const voiceFactSummary = document.getElementById("voice-fact-summary");
  const voiceFactReasoning = document.getElementById("voice-fact-reasoning");
  const voiceSourcesContainer = document.getElementById("voice-sources-container");

  let currentAudioBlob = null;
  let currentAudioFileName = "audio_chunk.wav";
  let mediaRecorder = null;
  let audioChunks = [];
  let recordTimer = null;
  let audioContext = null;
  let analyserNode = null;
  let visualizerAnimId = null;

  // ------------------------------------------------------------------------
  // LOAD PRESET SAMPLES
  // ------------------------------------------------------------------------
  async function loadPresetSamples() {
    try {
      const res = await fetch("/api/audio/samples");
      const data = await res.json();
      if (!data.samples || data.samples.length === 0) return;

      samplesContainer.innerHTML = "";
      data.samples.forEach(s => {
        const card = document.createElement("div");
        card.className = "sample-card";
        
        let badgeClass = "fact";
        if (s.tag.includes("Fraud")) badgeClass = "fraud";
        else if (s.tag.includes("Scam")) badgeClass = "utility";

        card.innerHTML = `
          <div>
            <div class="sample-top">
              <span class="sample-badge ${badgeClass}">${escapeHtml(s.tag)}</span>
              <span class="sample-lang">${escapeHtml(s.lang)}</span>
            </div>
            <div class="sample-title">${escapeHtml(s.title)}</div>
            <div class="sample-desc">${escapeHtml(s.description)}</div>
          </div>
          <div class="sample-action">
            <span>⚡ Test This Audio</span>
            <span>➔</span>
          </div>
        `;

        card.addEventListener("click", async () => {
          document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
          card.classList.add("active");
          await selectPresetAudio(s.url, s.filename);
        });

        samplesContainer.appendChild(card);
      });
    } catch (err) {
      console.warn("Failed to load audio samples:", err);
    }
  }

  async function selectPresetAudio(url, filename) {
    try {
      voiceBtnText.textContent = "Loading Audio Sample...";
      analyzeVoiceBtn.disabled = true;

      const res = await fetch(url);
      const blob = await res.blob();
      currentAudioBlob = blob;
      currentAudioFileName = filename;

      // Show preview
      audioPlayer.src = URL.createObjectURL(blob);
      audioPreviewWrapper.style.display = "flex";

      // Update file info
      selectedFileInfo.style.display = "flex";
      fileNameText.textContent = filename;

      analyzeVoiceBtn.disabled = false;
      voiceBtnText.textContent = "Transcribe & Analyze Chunk";
    } catch (e) {
      console.error(e);
      alert("Could not load preset audio.");
    }
  }

  loadPresetSamples();

  // ------------------------------------------------------------------------
  // FILE INPUT & DRAG AND DROP
  // ------------------------------------------------------------------------
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

  function handleAudioFile(file) {
    currentAudioBlob = file;
    currentAudioFileName = file.name;

    selectedFileInfo.style.display = "flex";
    fileNameText.textContent = file.name;

    audioPlayer.src = URL.createObjectURL(file);
    audioPreviewWrapper.style.display = "flex";

    analyzeVoiceBtn.disabled = false;
    voiceBtnText.textContent = "Transcribe & Analyze Chunk";
  }

  clearFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    currentAudioBlob = null;
    currentAudioFileName = "";
    audioFileInput.value = "";
    selectedFileInfo.style.display = "none";
    audioPreviewWrapper.style.display = "none";
    audioPlayer.src = "";
    analyzeVoiceBtn.disabled = true;
  });

  // ------------------------------------------------------------------------
  // LIVE MICROPHONE RECORDING (1 MINUTE MAX CHUNK)
  // ------------------------------------------------------------------------
  const MAX_RECORD_SECONDS = 60; // 1 minute maximum
  let elapsedSeconds = 0;
  const recordProgressFill = document.getElementById("record-progress-fill");
  const autoDetectCheckbox = document.getElementById("auto-detect-checkbox");
  const segmentsSection = document.getElementById("segments-section");
  const segmentsList = document.getElementById("segments-list");

  function formatMMSS(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  micBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      elapsedSeconds = 0;

      // Init audio visualizer
      setupAudioVisualizer(stream);

      // MediaRecorder
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
        currentAudioFileName = `mic_1min_chunk_${Date.now()}.${extension}`;

        audioPlayer.src = URL.createObjectURL(currentAudioBlob);
        audioPreviewWrapper.style.display = "flex";

        selectedFileInfo.style.display = "flex";
        fileNameText.textContent = `Live Mic Audio (${formatMMSS(elapsedSeconds)} / 1 min max)`;

        analyzeVoiceBtn.disabled = false;
        voiceBtnText.textContent = "Transcribe & Analyze Chunk";

        stream.getTracks().forEach(t => t.stop());
        cleanupVisualizer();

        // Auto-detect immediately if checkbox checked!
        if (autoDetectCheckbox && autoDetectCheckbox.checked && elapsedSeconds > 0) {
          console.log("Auto-detect triggered after recording completion.");
          setTimeout(() => {
            handleVoiceAnalysis();
          }, 300);
        }
      };

      mediaRecorder.start();
      micBtn.classList.add("recording");
      micIcon.textContent = "⏹️";
      micLabel.textContent = "Recording Audio...";
      micHint.textContent = "Click stop anytime or auto-stops at 1 minute (60s max)";

      timerDisplay.textContent = `00:00 / 01:00`;
      if (recordProgressFill) recordProgressFill.style.width = "0%";

      recordTimer = setInterval(() => {
        elapsedSeconds += 1;
        timerDisplay.textContent = `${formatMMSS(elapsedSeconds)} / 01:00`;
        
        if (recordProgressFill) {
          const pct = Math.min(100, (elapsedSeconds / MAX_RECORD_SECONDS) * 100);
          recordProgressFill.style.width = `${pct}%`;
        }

        if (elapsedSeconds >= MAX_RECORD_SECONDS) {
          console.log("Max 1 minute reached. Auto-stopping recording.");
          stopRecording();
        }
      }, 1000);

    } catch (err) {
      console.error("Mic access denied or error:", err);
      alert("Microphone access failed: " + err.message + "\nPlease make sure microphone permission is allowed or use the Audio Presets/Uploads.");
    }
  }

  function stopRecording() {
    if (recordTimer) {
      clearInterval(recordTimer);
      recordTimer = null;
    }
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    micBtn.classList.remove("recording");
    micIcon.textContent = "🎙️";
    micLabel.textContent = "Audio Captured (1 Min Max)";
    micHint.textContent = "Ready to verify with Whisper + Crawl4AI";
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

        ctx.fillStyle = "rgba(250, 250, 250, 0.6)";
        ctx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);

        const barWidth = (visualizerCanvas.width / bufferLength) * 1.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = Math.max(2, (dataArray[i] / 255) * visualizerCanvas.height);

          // ElevenLabs warm near-black ink with subtle intensity
          ctx.fillStyle = "#292524";
          ctx.beginPath();
          ctx.roundRect(x, visualizerCanvas.height - barHeight, barWidth - 1.5, barHeight, 2);
          ctx.fill();

          x += barWidth;
        }

      }

      draw();
    } catch (e) {
      console.warn("Visualizer failed to init:", e);
    }
  }

  function cleanupVisualizer() {
    if (visualizerAnimId) cancelAnimationFrame(visualizerAnimId);
    if (audioContext && audioContext.state !== "closed") audioContext.close();
    const ctx = visualizerCanvas.getContext("2d");
    ctx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
  }

  // ------------------------------------------------------------------------
  // ANALYZE AUDIO CHUNK SUBMISSION
  // ------------------------------------------------------------------------
  analyzeVoiceBtn.addEventListener("click", handleVoiceAnalysis);

  async function handleVoiceAnalysis() {
    if (!currentAudioBlob) {
      alert("Please record or select an audio chunk first.");
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

  function setVoiceLoading(isLoading) {
    if (isLoading) {
      analyzeVoiceBtn.disabled = true;
      voiceBtnText.textContent = "Transcribing & Grounding Evidence...";
      voiceBtnIcon.innerHTML = `<span class="spinner"></span>`;
      voiceLoadingBox.style.display = "block";
      voiceResultsBox.style.display = "none";
      voiceLoadingBox.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      analyzeVoiceBtn.disabled = false;
      voiceBtnText.textContent = "Transcribe & Analyze Chunk";
      voiceBtnIcon.textContent = "⚡";
      voiceLoadingBox.style.display = "none";
    }
  }

  function cycleVoiceLoadingMessages() {
    const msgs = [
      { status: "Transcribing with Faster-Whisper...", detail: "Running local int8 CPU inference on 4-second audio chunk" },
      { status: "Detecting Audio Language & Keywords...", detail: "Identifying Hindi / English / Hinglish speech patterns" },
      { status: "Evaluating Scam & Threat Matrix...", detail: "Analyzing psychological urgency, financial coercion, and fraud signals" },
      { status: "Crawl4AI Live Web Grounding...", detail: "Scraping authoritative sources & advisories via Playwright Chromium" },
      { status: "Synthesizing Defense Intelligence...", detail: "Finalizing risk score and emergency advisory directive" }
    ];
    let i = 0;
    return setInterval(() => {
      i = (i + 1) % msgs.length;
      voiceLoadingStatus.textContent = msgs[i].status;
      voiceLoadingDetail.textContent = msgs[i].detail;
    }, 2500);
  }

  function renderVoiceResults(data) {
    const t = data.transcription || {};
    const threat = data.threat || {};
    const fact = data.fact_check || {};
    const sources = data.sources || [];

    // 1. Transcription Metadata
    langPill.textContent = `${t.language || "Unknown"} (${t.language_code || ""})`;
    latencyPill.textContent = `IndicConformer Latency: ${t.inference_sec || 0}s`;
    durationPill.textContent = `Audio Duration: ${t.duration || 0}s`;
    
    if (t.transcript) {
      const hindiText = t.hindi_text || t.transcript;
      const hinglishText = t.hinglish_text || "";
      if (hinglishText && hinglishText !== hindiText) {
        transcribedTextBox.innerHTML = `
          <div style="margin-bottom: 8px;"><strong>Hindi Transcript:</strong> "${escapeHtml(hindiText)}"</div>
          <div style="color: #6ee7b7; border-top: 1px dashed rgba(255,255,255,0.15); padding-top: 8px;"><strong>Hinglish:</strong> "${escapeHtml(hinglishText)}"</div>
        `;
      } else {
        transcribedTextBox.textContent = `"${hindiText}"`;
      }
    } else {
      transcribedTextBox.textContent = "No intelligible speech detected in audio.";
    }

    // Render speech segments timeline for 1-minute audio
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

    // 2. Threat Hero
    const level = (threat.threat_level || "SAFE").toUpperCase();
    const score = Math.min(100, Math.max(0, threat.risk_score || 0));

    threatHero.className = "threat-hero";
    if (level.includes("CRITICAL") || score >= 75) {
      threatHero.classList.add("critical");
      threatIcon.textContent = "🚨";
      threatText.textContent = "CRITICAL SCAM DETECTED";
      scoreMeterFill.style.backgroundColor = "#f43f5e";
    } else if (level.includes("SUSPICIOUS") || score >= 40) {
      threatHero.classList.add("suspicious");
      threatIcon.textContent = "⚠️";
      threatText.textContent = "SUSPICIOUS COERCION / CLAIM";
      scoreMeterFill.style.backgroundColor = "#f59e0b";
    } else {
      threatHero.classList.add("safe");
      threatIcon.textContent = "🛡️";
      threatText.textContent = "SAFE / NORMAL SPEECH";
      scoreMeterFill.style.backgroundColor = "#10b981";
    }

    scamTypeTag.textContent = threat.scam_type || "Normal Conversation";
    threatScore.textContent = `${score} / 100`;
    scoreMeterFill.style.width = `${score}%`;

    // 3. Emergency Directive
    if (threat.action_advice) {
      adviceText.textContent = threat.action_advice;
      emergencyAdviceBox.style.display = "block";
    } else {
      emergencyAdviceBox.style.display = "none";
    }

    // 4. Red Flags
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

    // 5. Fact Grounding
    if (fact && fact.verdict) {
      voiceFactSection.style.display = "block";
      const v = fact.verdict.toUpperCase();
      voiceFactVerdict.textContent = v;
      voiceFactConfidence.textContent = `Confidence: ${fact.confidence || 0}%`;

      if (v.includes("SUPPORTED")) {
        voiceFactVerdict.style.backgroundColor = "rgba(16, 185, 129, 0.2)";
        voiceFactVerdict.style.color = "#34d399";
      } else if (v.includes("CONTRADICTED") || v.includes("FRAUD")) {
        voiceFactVerdict.style.backgroundColor = "rgba(244, 63, 94, 0.2)";
        voiceFactVerdict.style.color = "#fb7185";
      } else {
        voiceFactVerdict.style.backgroundColor = "rgba(245, 158, 11, 0.2)";
        voiceFactVerdict.style.color = "#fbbf24";
      }

      voiceFactSummary.textContent = fact.summary || "";
      voiceFactReasoning.textContent = fact.reasoning || "";
    } else {
      voiceFactSection.style.display = "none";
    }

    // 6. Sources list
    voiceSourcesContainer.innerHTML = "";
    if (sources.length > 0) {
      sources.forEach(src => {
        const card = document.createElement("div");
        card.className = "source-card";
        card.innerHTML = `
          <div class="source-header">
            <a href="${src.url}" target="_blank" rel="noopener noreferrer" class="source-title">
              ${escapeHtml(src.title || src.url)} ↗
            </a>
            <div class="source-meta">
              <span>Status: ${src.status}</span>
              <span>•</span>
              <span>Scraped: ${src.scraped_chars.toLocaleString()} chars</span>
            </div>
          </div>
          <div class="source-snippet">${escapeHtml(src.snippet || "Clean markdown content scraped.")}</div>
        `;
        voiceSourcesContainer.appendChild(card);
      });
    } else {
      voiceSourcesContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;">No external web sources were needed for this segment.</div>`;
    }

    voiceResultsBox.style.display = "block";
    voiceResultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------------
  // TAB 2: TEXT FACT CHECKER LOGIC
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

  // Preset pills for text tab
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      claimInput.value = btn.getAttribute("data-claim") || "";
      urlInput.value = btn.getAttribute("data-urls") || "";
      claimInput.focus();
    });
  });

  verifyTextBtn.addEventListener("click", handleTextVerification);

  async function handleTextVerification() {
    const claim = claimInput.value.trim();
    if (!claim) {
      alert("Please enter a claim or statement to fact-check.");
      claimInput.focus();
      return;
    }

    const rawUrls = urlInput.value.trim();
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

  function setTextLoading(isLoading) {
    if (isLoading) {
      verifyTextBtn.disabled = true;
      textBtnText.textContent = "Crawl4AI Scraping & Verifying...";
      textBtnIcon.innerHTML = `<span class="spinner"></span>`;
      textLoadingBox.style.display = "block";
      textResultsBox.style.display = "none";
      textLoadingBox.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      verifyTextBtn.disabled = false;
      textBtnText.textContent = "Detect & Verify Fact";
      textBtnIcon.textContent = "⚡";
      textLoadingBox.style.display = "none";
    }
  }

  function cycleTextLoadingMessages() {
    const msgs = [
      { status: "Launching Crawl4AI Headless Chromium...", detail: "Spinning up Playwright browser instances without third-party APIs" },
      { status: "Scraping Target Web Pages...", detail: "Navigating directly to live web pages and extracting clean markdown content" },
      { status: "Processing Page Content & Metadata...", detail: "Parsing DOM, cleaning boilerplate, and extracting factual data" },
      { status: "Fact-Checking with Gemini...", detail: "Evaluating claim against scraped evidence with strict reality-grounding" }
    ];
    let idx = 0;
    return setInterval(() => {
      idx = (idx + 1) % msgs.length;
      textLoadingStatus.textContent = msgs[idx].status;
      textLoadingDetail.textContent = msgs[idx].detail;
    }, 2600);
  }

  function renderTextResults(data) {
    const verdict = (data.verdict || "UNVERIFIABLE").toUpperCase();
    const confidence = Math.round(data.confidence || 0);

    textVerdictHero.className = "verdict-hero";
    textVerdictBadge.className = "verdict-badge";

    let icon = "❓";
    let statusClass = "unverifiable";

    if (verdict.includes("SUPPORTED") || verdict.includes("TRUE")) {
      icon = "✅";
      statusClass = "supported";
    } else if (verdict.includes("CONTRADICTED") || verdict.includes("FALSE") || verdict.includes("FRAUD")) {
      icon = "❌";
      statusClass = "contradicted";
    }

    textVerdictHero.classList.add(statusClass);
    textVerdictBadge.classList.add(statusClass);
    textVerdictIcon.textContent = icon;
    textVerdictText.textContent = verdict;

    textConfidenceVal.textContent = `${confidence}%`;
    textConfidenceFill.style.width = `${confidence}%`;
    textConfidenceFill.style.backgroundColor = 
      statusClass === "supported" ? "#10b981" :
      statusClass === "contradicted" ? "#f43f5e" : "#f59e0b";

    textSummaryContent.textContent = data.summary || "No summary provided.";
    textReasoningContent.textContent = data.reasoning || "No in-depth reasoning available.";

    // Key facts
    textFactsContainer.innerHTML = "";
    if (data.key_facts && data.key_facts.length > 0) {
      document.getElementById("text-key-facts-section").style.display = "block";
      data.key_facts.forEach(f => {
        const item = document.createElement("div");
        item.className = "fact-item";
        item.innerHTML = `
          <div class="fact-text">${escapeHtml(f.fact)}</div>
          <div class="fact-source">Source: ${escapeHtml(f.source || "Web Scrape")}</div>
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
        card.className = "source-card";
        card.innerHTML = `
          <div class="source-header">
            <a href="${src.url}" target="_blank" rel="noopener noreferrer" class="source-title">
              ${escapeHtml(src.title || src.url)} ↗
            </a>
            <div class="source-meta">
              <span>Status: ${src.status}</span>
              <span>•</span>
              <span>Scraped: ${src.scraped_chars.toLocaleString()} chars</span>
            </div>
          </div>
          <div class="source-snippet">${escapeHtml(src.snippet || "No preview snippet available.")}</div>
        `;
        textSourcesContainer.appendChild(card);
      });
    } else {
      textSourcesContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 0.9rem;">No source records returned.</div>`;
    }

    textResultsBox.style.display = "block";
    textResultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
