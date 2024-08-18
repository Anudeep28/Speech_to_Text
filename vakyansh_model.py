import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import pyaudio
import numpy as np
import re

# Load the model and processor
model_name = "Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define audio stream parameters
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)

print("Starting real-time transcription... Press Ctrl+C to stop.")

final_transcript = []
try:
    while True:
        # Read audio data from the stream
        data = stream.read(1024)
        # Convert bytes data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Normalize the audio data
        audio_data = audio_data / 32768.0
        
        # Convert numpy array to tensor
        # Convert numpy array to tensor
        input_values = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)["input_values"].to("cuda")
        #input_values = processor(torch.tensor([data]), return_tensors="pt", padding=True).input_values

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        print(predicted_ids.shape)
        transcription = processor.decode(predicted_ids[0])
        transcription = re.sub(r'<s>', '', transcription)
        final_transcript.append(transcription)
        print(f"Transcription: {' '.join(i for i in final_transcript)}")

except KeyboardInterrupt:
    print("Stopping transcription...")

finally:
    # Close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(final_transcript)
