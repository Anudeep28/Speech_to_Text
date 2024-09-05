import torch
import torchaudio
import whisper


# Example of using beam search for inference
def transcribe_with_beam_search(model_path: str, audio, beam_size: int=5, best_of: int=3):
    # Load the whsper model from the path
    model = whisper.load_model(model_path)
    # The OpenAI Whisper library uses beam search by default
    # We just need to specify the beam size
    waveform, sample_rate = torchaudio.load(audio)
    waveform = whisper.pad_or_trim(waveform)
    
    model.eval()
    with torch.inference_mode():
        result = model.transcribe(waveform.squeeze(0), beam_size=beam_size, best_of=best_of)
        return result["text"]