import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer
import get_data, data_setup, initialize_whisper_model, split_datasets, whisper_engine, utils

# Constants
SPLIT_REQUIRED = False
SPLIT_SIZE = 0.8
NUM_WORKERS = min(2, os.cpu_count() - 1)  # Use at most 2 workers, leaving 1 CPU free
BATCH_SIZE = 32  # Increased batch size for better GPU utilization
MODEL_TYPE = "tiny.en"
LANGUAGE = "en"
TASK = "transcribe"
TARGET_DIR = "whisper_models"
MODEL_NAME = "whisper_model_version_1.pth"
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4  # Slightly increased learning rate

# Data paths
AUDIO_DIR = os.path.join(os.getcwd(), 'dataset_anu', 'audio')
TRANSCRIPT_DIR = os.path.join(os.getcwd(), 'dataset_anu', 'transcript')

def main():
    # Load and prepare data
    data = get_data.load_data(audio_dir=AUDIO_DIR, transcript_dir=TRANSCRIPT_DIR)
    compiled_data = pd.DataFrame(data)

    # Set up dataset and dataloader
    dataset = data_setup.WhisperDataset(compiled_data, AUDIO_DIR, augment_both=True)
    
    if SPLIT_REQUIRED:
        train, test = split_datasets.split_dataset(dataset=dataset, split_size=SPLIT_SIZE)
        train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_setup.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
        test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=data_setup.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_setup.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
        train_dataloader = test_dataloader = dataloader

    # Initialize model and tokenizer
    model, decoder_start_token_id = initialize_whisper_model.initialize_whisper(model_type=MODEL_TYPE, multilingual=False, language=LANGUAGE, task=TASK)
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{MODEL_TYPE}", language=LANGUAGE, task=TASK)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Train the model
    whisper_engine.train(model=model,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,
                         audio_type="augmented_log_mel_spec",
                         loss_fn=loss_fn,
                         optimizer=optimizer,
                         epochs=NUM_EPOCHS,
                         device=device,
                         decoder_start_token_id=decoder_start_token_id,
                         tokenizer=tokenizer,
                         scaler=scaler)

    # Save the model
    utils.save_model(model=model, target_dir=TARGET_DIR, model_name=MODEL_NAME)

if __name__ == "__main__":
    main()
