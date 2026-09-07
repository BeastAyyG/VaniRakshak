"""
Transcript Buffer for VaniRakshak Audio Pipeline

Maintains sequential 4-second chunk transcripts with precise timestamps,
preventing loss of temporal evidence for downstream fraud detection and claim extraction.
"""

from typing import List, Dict, Any, Optional
from .types import TranscriptionResult


class TranscriptBuffer:
    """
    Buffers sequential audio chunk transcription results and generates
    both structured chunk lists with timestamps and aggregated full transcripts.
    """

    def __init__(self):
        self._chunks: List[TranscriptionResult] = []

    def add_chunk_result(self, result: TranscriptionResult) -> None:
        """Appends a new chunk transcription result to the chronological buffer."""
        self._chunks.append(result)

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        """Returns structured list of transcribed chunks with start/end timestamps."""
        serialized = []
        for c in self._chunks:
            serialized.append({
                "start": round(c.start_time, 2) if c.start_time is not None else 0.0,
                "end": round(c.end_time, 2) if c.end_time is not None else 0.0,
                "text": c.text,
                "hindi_text": c.text,
                "hinglish_text": c.hinglish_text or "",
                "confidence": c.confidence,
                "rtf": c.rtf
            })
        return serialized

    @property
    def full_hindi_text(self) -> str:
        """Produces aggregated, coherent Hindi transcript across all received chunks."""
        texts = [c.text.strip() for c in self._chunks if c.text and c.text.strip()]
        return " ".join(texts)

    @property
    def full_hinglish_text(self) -> str:
        """Produces aggregated, coherent Hinglish transcript across all received chunks."""
        texts = [c.hinglish_text.strip() for c in self._chunks if c.hinglish_text and c.hinglish_text.strip()]
        return " ".join(texts)

    @property
    def total_duration(self) -> float:
        """Calculates total duration across buffered chunks."""
        if not self._chunks:
            return 0.0
        start = self._chunks[0].start_time or 0.0
        end = self._chunks[-1].end_time or 0.0
        return max(0.0, end - start)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire transcript buffer."""
        return {
            "chunks": self.chunks,
            "hindi_text": self.full_hindi_text,
            "hinglish_text": self.full_hinglish_text,
            "total_chunks": len(self._chunks),
            "total_duration_seconds": round(self.total_duration, 2)
        }

    def clear(self) -> None:
        """Clears all buffered chunks."""
        self._chunks.clear()
