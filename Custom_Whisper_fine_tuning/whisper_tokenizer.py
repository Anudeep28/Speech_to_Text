import whisper
from whisper.tokenizer import get_tokenizer

# Load the Whisper model
model = whisper.load_model("large")

# Load and preprocess audio
audio_file = "audio.mp3"
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio)

# Create log-Mel spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Define decoding options without a language model
decode_options = whisper.DecodingOptions(
    language="en",  # Specify the language if known
    beam_size=5,
    best_of=3,
    temperature=0.2
)

# Transcribe using Whisper
result = model.transcribe(audio_file, **decode_options)

# Print the recognized text
print("Transcription:", result["text"])

# Get the token vocabulary from the Whisper model
tokenizer = get_tokenizer(multilingual=False, language='en', task='transcribe')

# Example: Encode a sample text to see the corresponding tokens
sample_text = "Hello, how are you?"
encoded_tokens = tokenizer.encode(sample_text)
decoded_tokens = tokenizer.decode(encoded_tokens)

print("Encoded Tokens:", encoded_tokens)
print("Decoded Tokens:", decoded_tokens)