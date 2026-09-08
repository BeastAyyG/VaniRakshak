# 🎙️ VaniRakshak Test Audio Directory

Place your Hindi or Hinglish audio files (`.wav`, `.mp3`, `.m4a`, `.flac`) here for testing the AI4Bharat IndicConformer ASR pipeline.

### Pre-loaded Audio Files:
- `hindi_sample.wav`: Test Case 1 — Native Hindi: *"आपका बैंक अकाउंट कल बंद हो जाएगा।"*
- `hinglish_sample.wav`: Test Case 2 — Hinglish Code-Switching: *"Sir aapka bank account kal suspend ho jayega."*
- `scam_sample.wav`: Test Case 3 — Real-world Scam Phrasing: *"Sir main SBI headquarters se bol raha hoon. Aapka KYC expire ho gaya hai."*
- `sample_bijli_hi.mp3`: Hindi Electricity Disconnection Scam Call.
- `sample_test_voice.wav`: Standard conversational sample.

### How to Run:
```bash
# Test specific audio with IndicConformer:
python scripts/test_indicconformer.py test_audio/hindi_sample.wav

# Or test with custom audio:
python scripts/test_indicconformer.py test_audio/your_audio.wav
```
