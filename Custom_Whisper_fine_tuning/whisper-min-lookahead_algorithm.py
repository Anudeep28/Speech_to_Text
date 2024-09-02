import torch
import whisper
import numpy as np
from typing import List, Tuple

def load_audio(file: str) -> np.ndarray:
    # This is a placeholder. In a real scenario, you'd use a library like librosa or soundfile
    # to load the audio file and return it as a numpy array
    return np.zeros(16000)  # Placeholder: 1 second of silence at 16kHz

def transcribe_with_timestamps(model: whisper.Whisper, audio: np.ndarray) -> List[dict]:
    # Transcribe audio using Whisper model
    result = model.transcribe(audio, word_timestamps=True)
    return result["segments"]

def min_lookahead(segments: List[dict], lookahead: float = 1.0) -> List[Tuple[float, str]]:
    words = []
    for segment in segments:
        words.extend(segment["words"])
    
    aligned_words = []
    buffer = []
    current_time = 0

    for word in words:
        while buffer and buffer[0]["end"] <= word["start"] + lookahead:
            earliest = buffer.pop(0)
            aligned_words.append((max(current_time, earliest["start"]), earliest["word"]))
            current_time = max(current_time, earliest["start"])
        
        buffer.append(word)
    
    # Flush remaining buffer
    for word in buffer:
        aligned_words.append((max(current_time, word["start"]), word["word"]))
        current_time = max(current_time, word["start"])
    
    return aligned_words

def main():
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Load audio file
    audio = load_audio("path/to/your/audio/file.wav")
    
    # Transcribe with timestamps
    segments = transcribe_with_timestamps(model, audio)
    
    # Apply Min Lookahead algorithm
    aligned_words = min_lookahead(segments, lookahead=1.0)
    
    # Print aligned words
    for timestamp, word in aligned_words:
        print(f"{timestamp:.2f}: {word}")

if __name__ == "__main__":
    main()
