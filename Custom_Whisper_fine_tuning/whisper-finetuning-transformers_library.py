import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Trainer, TrainingArguments
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, spectrograms, transcriptions, processor):
        self.spectrograms = spectrograms
        self.transcriptions = transcriptions
        self.processor = processor

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        input_features = torch.from_numpy(self.spectrograms[idx]).float()
        labels = self.processor(text=self.transcriptions[idx], return_tensors="pt").input_ids.squeeze()
        return {"input_features": input_features, "labels": labels}

def main():
    # Load your preprocessed data
    spectrograms = np.load("path/to/your/spectrograms.npy")
    with open("path/to/your/transcriptions.txt", "r") as f:
        transcriptions = f.readlines()

    # Initialize the Whisper model and processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Create custom dataset
    dataset = CustomDataset(spectrograms, transcriptions, processor)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        save_steps=1000,
        logging_steps=100,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_whisper")

    # Example of using beam search for inference
    def transcribe_with_beam_search(audio, beam_size=5):
        input_features = processor(audio, return_tensors="pt").input_features
        
        # Generate with beam search
        outputs = model.generate(
            input_features,
            num_beams=beam_size,
            early_stopping=True,
            max_length=256  # Adjust as needed
        )
        
        return processor.batch_decode(outputs, skip_special_tokens=True)

    # Example usage (assuming you have a test audio file)
    test_audio = np.load("path/to/test_audio.npy")
    transcription = transcribe_with_beam_search(test_audio, beam_size=5)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
