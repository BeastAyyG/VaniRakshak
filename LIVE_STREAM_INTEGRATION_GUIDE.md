# VaniRakshak — Live Voice Call Stream Integration Architecture & Guide
**Document Version:** 2.4.0  
**Target:** Mobile Main App (Android / iOS / VoIP / Telephony PBX) to VaniRakshak Engine

---

## 1. Executive Overview

This guide specifies how your **main mobile application** (or telephony PBX gateway) connects to VaniRakshak to stream **live voice call recordings in real-time**. As the caller and user converse, audio buffers are streamed continuously over a low-latency connection, evaluated incrementally against scam & fraud heuristics, and updated live on the VaniRakshak security dashboard.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────────────────────┘

  [ Mobile Device / Main App ]
  ┌──────────────────────────┐
  │ Active Phone Call        │
  │ AudioRecord / CallKit    │
  │ (16kHz, 16-bit PCM Mono) │
  └─────────────┬────────────┘
                │
                │ Binary Audio Stream (WebSocket or Chunked REST)
                ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ VaniRakshak Streaming Engine (Port 8084)                                  │
  │                                                                           │
  │  /ws/live-call-stream (WebSocket)   /api/stream/call-chunk (REST fallback)│
  │         │                                      │                          │
  │         ▼                                      ▼                          │
  │  ┌───────────────────────┐             ┌───────────────────────┐          │
  │  │ Sliding Audio Buffer  │             │ Multi-Lingual ASR     │          │
  │  │ (In-Memory Ring)      │────────────▶│ (Hindi/English/Hing)  │          │
  │  └───────────────────────┘             └───────────┬───────────┘          │
  │                                                    │ Incremental Text     │
  │                                                    ▼                      │
  │                                        ┌───────────────────────┐          │
  │                                        │ Scam Threat Matrix    │          │
  │                                        │ & Coercion Classifier │          │
  │                                        └───────────┬───────────┘          │
  │                                                    │ Threat Score & Flags │
  │                                                    ▼                      │
  │                                        ┌───────────────────────┐          │
  │                                        │ Live Ground-Truth     │          │
  │                                        │ Web Evidence Search   │          │
  │                                        └───────────┬───────────┘          │
  └────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                           ┌───────────────────────────┴──────────────────────────┐
                           ▼                                                      ▼
             [ VaniRakshak Web Dashboard ]                          [ Main App Defense Alert ]
             • Real-time rolling transcript                         • Heads-up warning overlay
             • Dynamic risk gauge (0-100)                           • "HANG UP IMMEDIATELY"
             • Emergency directive banner                           • Auto-mute / Fraud trigger
```

---

## 2. Ingestion Protocols Supported

VaniRakshak exposes **two complementary streaming interfaces**:

| Protocol | Endpoint | Best For | Latency |
| :--- | :--- | :--- | :--- |
| **WebSocket** *(Recommended)* | `ws://<host>:8084/ws/live-call-stream` | Real-time bi-directional streaming, live word-by-word alerts | **< 300ms** |
| **Chunked REST** *(Fallback)* | `POST /api/stream/call-chunk` | Mobile background services, high-latency or unstable networks | **~ 1.5s** |

---

## 3. Protocol 1: WebSocket Real-Time Stream (Recommended)

### 3.1 Connection Handshake
- **URL:** `ws://<SERVER_IP>:8084/ws/live-call-stream?call_id=<UNIQUE_UUID>&caller=<CALLER_PHONE>`
- **Sample URL:** `ws://127.0.0.1:8084/ws/live-call-stream?call_id=call_987214981&caller=%2B919876543210`

Upon opening, the server transmits a handshake confirmation:
```json
{
  "event": "stream_connected",
  "call_id": "call_987214981",
  "status": "ready",
  "caller": "+919876543210",
  "threat_score": 0,
  "threat_level": "SAFE"
}
```

### 3.2 Streaming Audio Data (Client ➔ Server)
The client continuously transmits raw binary frames containing audio samples:
- **Sample Rate:** 16,000 Hz (16 kHz)
- **Channels:** 1 (Mono)
- **Encoding:** 16-bit Linear PCM (Signed 16-bit Little Endian) or WebM/Opus chunks.
- **Packet Cadence:** Send chunks every **500ms – 1000ms** (~16,000 to 32,000 bytes per frame).

### 3.3 Server Feedback Events (Server ➔ Client)
As speech is detected and processed, the server pushes JSON frames back to the client:

#### A) Incremental Threat Update (`stream_update`)
```json
{
  "event": "stream_update",
  "call_id": "call_987214981",
  "incremental_text": "Aapka bank account freeze kiya gaya hai, kripya 25000 transfer karein.",
  "full_transcript": "Hello kya aap sun rahe hain? Mumbai police se bol rahe hain. Aapka bank account freeze kiya gaya hai, kripya 25000 transfer karein.",
  "threat_score": 92,
  "threat_level": "CRITICAL_SCAM",
  "scam_type": "Bank KYC & Digital Arrest Fraud",
  "action_advice": "HANG UP IMMEDIATELY. Legitimate banks and police never request funds into security escrow accounts over the phone.",
  "red_flags": [
    "Threat of immediate account freeze",
    "Impersonation of law enforcement",
    "Demand for urgent financial transfer"
  ],
  "duration_sec": 14.5,
  "is_completed": false
}
```

#### B) End of Call (`call_completed`)
When the call terminates, the client sends `{"event": "end_call"}` or disconnects. The server returns the final forensic dossier:
```json
{
  "event": "call_completed",
  "call_id": "call_987214981",
  "duration_sec": 48.2,
  "full_transcript": "...",
  "threat_score": 95,
  "threat_level": "CRITICAL_SCAM",
  "is_completed": true
}
```

---

## 4. Mobile Main App Implementation Examples

### 4.1 Android Implementation (Kotlin + OkHttp)

```kotlin
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import android.media.AudioRecord
import android.media.AudioFormat
import android.media.MediaRecorder

class VaniRakshakCallStreamer(private val serverUrl: String) {
    private val client = OkHttpClient()
    private var webSocket: WebSocket? = null
    private var isStreaming = false

    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT) * 2

    fun startCallStream(callId: String, onAlertReceived: (threatScore: Int, advice: String) -> Unit) {
        val request = Request.Builder()
            .url("$serverUrl/ws/live-call-stream?call_id=$callId")
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                isStreaming = true
                startAudioRecordingThread()
            }

            override fun onMessage(ws: WebSocket, text: String) {
                val json = org.json.JSONObject(text)
                if (json.optString("event") == "stream_update") {
                    val score = json.optInt("threat_score", 0)
                    val advice = json.optString("action_advice", "")
                    if (score >= 75) {
                        // Trigger immediate Heads-Up Scam Alert on the user's phone!
                        onAlertReceived(score, advice)
                    }
                }
            }
        })
    }

    private fun startAudioRecordingThread() {
        Thread {
            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE
            )
            val buffer = ByteArray(BUFFER_SIZE)
            audioRecord.startRecording()

            while (isStreaming) {
                val read = audioRecord.read(buffer, 0, buffer.size)
                if (read > 0) {
                    // Stream raw audio packet directly to VaniRakshak server
                    webSocket?.send(buffer.copyOf(read).toByteString())
                }
            }
            audioRecord.stop()
            audioRecord.release()
        }.start()
    }

    fun stopCallStream() {
        isStreaming = false
        webSocket?.send("{\"event\": \"end_call\"}")
        webSocket?.close(1000, "Call Ended")
    }
}
```

---

### 4.2 iOS Implementation (Swift + AVAudioEngine)

```swift
import Foundation
import AVFoundation

class VaniRakshakCallStreamer: NSObject, URLSessionWebSocketDelegate {
    private var webSocketTask: URLSessionWebSocketTask?
    private let audioEngine = AVAudioEngine()

    func startStreaming(callId: String, serverHost: String) {
        guard let url = URL(string: "ws://\(serverHost):8084/ws/live-call-stream?call_id=\(callId)") else { return }
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: OperationQueue())
        webSocketTask = session.webSocketTask(with: url)
        webSocketTask?.resume()

        listenForThreatUpdates()
        startAudioTap()
    }

    private func startAudioTap() {
        let inputNode = audioEngine.inputNode
        let recordingFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!

        inputNode.installTap(onBus: 0, bufferSize: 2048, format: recordingFormat) { [weak self] (buffer, time) in
            guard let self = self else { return }
            let channelData = buffer.int16ChannelData?[0]
            let length = Int(buffer.frameLength) * MemoryLayout<Int16>.size
            if let data = channelData {
                let audioData = Data(bytes: data, count: length)
                self.webSocketTask?.send(.data(audioData)) { error in
                    if let error = error { print("Stream send error: \(error)") }
                }
            }
        }
        try? audioEngine.start()
    }

    private func listenForThreatUpdates() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                if case .string(let text) = message {
                    // Parse threat score and broadcast local alert if needed
                    print("Received threat update: \(text)")
                }
                self?.listenForThreatUpdates()
            case .failure(let error):
                print("WebSocket receive error: \(error)")
            }
        }
    }

    func stopStreaming() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        webSocketTask?.cancel(with: .goingAway, reason: nil)
    }
}
```

---

## 5. Protocol 2: Chunked REST Ingestion (Fallback)

For networks or OS environments where persistent WebSockets are throttled in the background, send sequential audio slices via HTTP:

- **Endpoint:** `POST http://<host>:8084/api/stream/call-chunk`
- **Content-Type:** `multipart/form-data`

### Parameters:
- `call_id` *(string, required)*: Unique identifier for the ongoing call.
- `chunk_index` *(integer, required)*: Zero-indexed incremental chunk number (`0, 1, 2...`).
- `is_final` *(boolean, optional)*: Set `true` when caller hangs up.
- `file` *(binary audio file, required)*: WAV, MP3, or WebM audio slice.

### Sample cURL Request:
```bash
curl -X POST "http://127.0.0.1:8084/api/stream/call-chunk" \
  -F "call_id=call_demo_1001" \
  -F "chunk_index=0" \
  -F "is_final=false" \
  -F "file=@chunk_00.wav"
```

### JSON Response:
```json
{
  "status": "success",
  "chunk_index": 0,
  "incremental_text": "This is Mumbai Police cyber cell.",
  "threat_score": 45,
  "threat_level": "SUSPICIOUS",
  "scam_type": "Digital Arrest Coercion",
  "action_advice": "Caller claims to be police. Verify badge and never transfer funds."
}
```

---

## 6. How It Appears on the VaniRakshak Website

When a call streams into the server:
1. **Live Stream Call Gateway** card automatically switches to `● Live Call Streaming Active`.
2. **Incremental Speech Feed** dynamically prints the spoken transcript as the caller speaks.
3. **Live Threat Assessment Gauge** shifts from 0% (Safe) to 90%+ (Critical Alert) in real time.
4. **Emergency Directive Alert** pops up with immediate instructions (e.g. *“Hang up immediately — 1930 Cyber Helpline”*).

Users can also test this immediately on the website by clicking **"Test Live Call Stream"**.
