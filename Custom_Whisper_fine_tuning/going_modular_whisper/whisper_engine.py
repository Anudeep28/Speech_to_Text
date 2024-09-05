"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# For temporary reason using tokenizer fromt he transformers library
# later i want to use the whisper tokenizer


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               audio_type: str,
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               decoder_start_token_id: int,
               tokenizer,
               scaler) -> float:
  """Trains a Whisper model for a single epoch.

  Turns a target model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    audio_type: type of audio formats available for training are -> 
        1. 'waveform': waveforms,
        2. 'mel_spectrogram': mel_specs,
        3. 'augmented_mel_spec':augmented_mel_spec,
        4. 'mfcc': mfccs,
        5. 'augmented_waveform': augmented_waveforms,
        6. 'augmented_log_mel_spec': augmented_log_mel_specs,
        7. 'transcript': transcripts
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss = 0

  # Loop through data loader data batches
  for idx, batch in enumerate(dataloader):
      
      # get the required data from the batch
      log_mel = batch[audio_type]
      # Squeeze to remove singleton dimensions
      log_mel = log_mel.squeeze(1)  # Change shape from [batch_size, channels, 80, 3000] to [batch_size, 80, 3000]

      # Get the transcripts
      # Tokenize transcripts
      text = tokenizer(batch["transcript"], return_tensors="pt", padding=True, truncation=True).input_ids
      # get the tokenized label sequences
      label_features = [{"input_ids": feature} for feature in text]
      # pad the labels to max length
      labels_batch = tokenizer.pad(label_features)
      # replace padding with -100 to ignore loss correctly
      labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
      # if bos token is appended in previous tokenization step,
      # cut bos token here as it's append later anyways
      if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
          labels = labels[:, 1:]

      # Getting the data to the correct device
      # To device
      log_mel = log_mel.to(device)
      labels = labels.to(device)

      # 1. Forward pass
      logits = model(log_mel,labels)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
      train_loss +=loss.item()

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # Backward pass with scaling
      if scaler is not None:
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
      else:
          # 4. Loss backward
          loss.backward()
          # 5. Optimizer step
          optimizer.step()

      # # Calculate and accumulate accuracy metric across all batches
      # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      # train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  #train_acc = train_acc / len(dataloader)
  return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              audio_type: str, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              decoder_start_token_id: int,
              tokenizer) -> float:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss = 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for idx, batch in enumerate(dataloader):
          # get the required data from the batch
          log_mel = batch[audio_type]
          # Squeeze to remove singleton dimensions
          log_mel = log_mel.squeeze(1)  # Change shape from [batch_size, channels, 80, 3000] to [batch_size, 80, 3000]

          # Get the transcripts
          # Tokenize transcripts
          text = tokenizer(batch["transcript"], return_tensors="pt", padding=True, truncation=True).input_ids
          # get the tokenized label sequences
          label_features = [{"input_ids": feature} for feature in text]
          # pad the labels to max length
          labels_batch = tokenizer.pad(label_features)
          # replace padding with -100 to ignore loss correctly
          labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
          # if bos token is appended in previous tokenization step,
          # cut bos token here as it's append later anyways
          if (labels[:, 0] == decoder_start_token_id).all().cpu().item():
              labels = labels[:, 1:]

          # Getting the data to the correct device
          # To device
          log_mel = log_mel.to(device)
          labels = labels.to(device)

          # 1. Forward pass
          logits = model(log_mel,labels)

          # 2. Calculate  and accumulate loss
          loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

          # loss
          test_loss += loss.item()

          # # Calculate and accumulate accuracy
          # test_pred_labels = test_pred_logits.argmax(dim=1)
          # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  #test_acc = test_acc / len(dataloader)
  return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          audio_type: str,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          decoder_start_token_id: int,
          tokenizer,
          scaler) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # check the given audio type matches the existing available types from dataloaders
  types = ['waveform','mel_spectrogram','augmented_mel_spec','mfcc','augmented_waveform','augmented_log_mel_spec']

  assert audio_type in types, "audio_type should be 'waveform'|'mel_spectrogram'|'augmented_mel_spec'|'mfcc'|'augmented_waveform'|'augmented_log_mel_spec'"

  
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss = train_step(model=model,
                                          dataloader=train_dataloader,
                                          audio_type=audio_type,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          decoder_start_token_id=decoder_start_token_id,
                                          tokenizer=tokenizer,
                                          scaler=scaler)
      test_loss = test_step(model=model,
                                          dataloader=test_dataloader,
                                          audio_type=audio_type,
                                          loss_fn=loss_fn,
                                          device=device,
                                          decoder_start_token_id=decoder_start_token_id,
                                          tokenizer=tokenizer)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          #f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          #f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      #results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      #results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results