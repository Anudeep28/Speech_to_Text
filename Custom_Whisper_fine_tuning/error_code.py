# Set up dataset and dataloader
# csv_file = 'path/to/your/metadata.csv'  # CSV file with columns: [audio_file_name, transcript]
# root_dir = 'path/to/your/audio/files'   # Directory containing audio files
# Load data
audio_dir = os.path.join(os.getcwd(), 'dataset_anu', 'audio')
transcript_dir = os.path.join(os.getcwd(), 'dataset_anu', 'transcript')
data = load_data(audio_dir, transcript_dir)

# Convert to pandas format
# Convert to DataFrame
compiled_data = pd.DataFrame(data)

dataset = WhisperDataset(compiled_data, audio_dir)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)


# Prepare the model for fine-tuning
model.train()

# Define optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for batch in dataloader:
        log_mel = batch["augmented_log_mel_spec"].to(device)
        #print(batch["transcript"])
        text = tokenizer(batch["transcript"], return_tensors="pt", padding=True, truncation=True).to(device)
        print(f"text shape: {text['input_ids'].shape}")

        # get the tokenized label sequences
        # label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = tokenizer.pad(text, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the channel dimension if it's not needed for wisper model when 
        # prociding a log mel tensor
        log_mel = log_mel.squeeze(1)  # Change shape from [10, 1, 80, 2401] to [10, 80, 2401]
        print(f"shape of log_mel: {log_mel.shape}")

        # Forward pass
        output = model(log_mel, labels)#text["input_ids"])
        loss = output.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    break