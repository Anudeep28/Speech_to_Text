import whisper
from pyctcdecode import build_ctcdecoder
import kenlm

# Load the Whisper model
model = whisper.load_model("large")

# Load and preprocess audio
audio_file = "audio.mp3"
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio)

# Create log-Mel spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Define decoding options
decode_options = whisper.DecodingOptions(
    language="en",
    beam_size=5,
    best_of=3,
    temperature=0.2
)

# Transcribe using Whisper with beam search
result = model.transcribe(audio_file, **decode_options)

# Print the initial transcription
print("Initial Transcription:", result["text"])

# Load the KenLM language model
lm_model_path = "path/to/your/kenlm_model.arpa"
lm_model = kenlm.Model(lm_model_path)

# Create a CTC decoder with KenLM rescoring
decoder = build_ctcdecoder(
    model.tokenizer.get_vocab(),
    lm_model=lm_model,
    alpha=0.5,
    beta=1.0
)

# Decode logits with KenLM rescoring
logits = model(audio_file)  # Get logits from the model
decoded_output = decoder.decode(logits)

# Print the re-scored transcription
print("Re-scored Transcription:", decoded_output)