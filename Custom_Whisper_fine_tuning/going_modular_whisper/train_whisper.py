# All the imports
import os
# Pandas
import pandas as pd
# torch
import torch
from torch.utils.data import DataLoader
# Going Modular imports
import get_data, data_setup, \
        visualize_audio, initialize_whisper_model, \
        split_datasets, whisper_engine, utils, \
        transcribe, optimized_get_dataset
# transformers imports
from transformers import WhisperTokenizer

# Setting up the dataset directories and all the required Variables
split_required = False
SPLIT_SIZE = 0.8
# activate the num workerd when working with lot of data
NUM_WORKERS = min(2, os.cpu_count() - 1)  # Use at most 2 workers, leaving 1 CPU free
BATCH_SIZE = 5
# Data Paths
audio_dir = os.path.join(os.getcwd(), 'dataset_anu', 'audio')
transcript_dir = os.path.join(os.getcwd(), 'dataset_anu', 'transcript')
# model requirements
model_type = "tiny.en"
language = "en"
task = "transcribe"
### Save the model to the disk with name
target_dir = "whisper_models"
model_name = "whisper_model_version_1_updated_code.pth"
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4  # Slightly increased learning rate
WEIGHT_DECAY = 0.01

# Getting the data into json format
data = get_data.load_data(audio_dir=audio_dir, transcript_dir=transcript_dir)

# Convert to pandas format
# Convert to DataFrame
compiled_data = pd.DataFrame(data)


# Setup the dataset with augmentation and dataloader for training
dataset = optimized_get_dataset.WhisperDataset(compiled_data, audio_dir, augment_both=True, signal_shift=False)
# If you dont need to split the dataset into trainig and testing
if split_required:
    # Split the dataset into training and testing
    train, test = split_datasets.split_dataset(dataset=dataset,
                                           split_size=SPLIT_SIZE)
    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=optimized_get_dataset.collate_fn)#, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=optimized_get_dataset.collate_fn)#, num_workers=NUM_WORKERS, pin_memory=True)
else:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=optimized_get_dataset.collate_fn)#, num_workers=NUM_WORKERS, pin_memory=True)#, num_workers=NUM_WORKERS, pin_memory=True)
    train_dataloader = dataloader
    test_dataloader = dataloader


# Initialize the model and toeknizer
# returns the model and decoder start token id
model, decoder_start_token_id = initialize_whisper_model.initialize_whisper(model_type=model_type,
                                                                            multilingual=False,
                                                                            language=language,
                                                                            task=task)


# Tokenizer fro pre-processing of the transcript before trining the model
# I have to see if the whisper librayr toeknizer works the same way it does for it

tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_name_or_path=f"openai/whisper-{model_type}", language=language,task=task)


# Setting up the Loss function and optimizer for the training of modle
# Have to learn from ketan doshi on more parameters for AdamW  optimizer

optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# Setup the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Use mixed precision training if available
scaler = torch.amp.GradScaler(device)

# Start training with help from engine.py
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


# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir=target_dir,
                 model_name=model_name)

