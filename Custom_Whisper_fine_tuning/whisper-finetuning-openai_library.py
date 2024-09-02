import torch
from torch.utils.data import Dataset, DataLoader
import whisper
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, spectrograms, transcriptions):
        self.spectrograms = spectrograms
        self.transcriptions = transcriptions

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return {"mel": torch.from_numpy(self.spectrograms[idx]).float(), "text": self.transcriptions[idx]}

def main():
    # Load your preprocessed data
    spectrograms = np.load("path/to/your/spectrograms.npy")
    with open("path/to/your/transcriptions.txt", "r") as f:
        transcriptions = f.readlines()

    # Initialize the Whisper model
    model = whisper.load_model("small")

    # Create custom dataset and dataloader
    dataset = CustomDataset(spectrograms, transcriptions)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Prepare the model for fine-tuning
    model.train()

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        for batch in dataloader:
            mel = batch["mel"]
            text = batch["text"]

            # Forward pass
            output = model(mel, text)
            loss = output.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_whisper_openai")

    # Example of using beam search for inference
    def transcribe_with_beam_search(audio, beam_size=5):
        # The OpenAI Whisper library uses beam search by default
        # We just need to specify the beam size
        result = model.transcribe(audio, beam_size=beam_size)
        return result["text"]

    # Example usage (assuming you have a test audio file)
    test_audio = "path/to/test_audio.wav"
    transcription = transcribe_with_beam_search(test_audio, beam_size=5)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
