import random
import torch
from torch.utils.data import DataLoader
import audio_utils

def visualize_audio(dataloader: DataLoader,
                    vis_audio: bool=False,
                    vis_transfd_audio: bool=False,
                    vis_spectrogram: bool=True,
                    vis_transfd_spectrogram:bool=False,
                    vis_mfcc: bool=False,
                    SAMPLE_RATE: int=16000) -> None:
    """
    Visualize audio data using a DataLoader.

    This function takes a DataLoader object containing audio data and visualizes
    the audio samples. It is assumed that the DataLoader yields tuples of
    (audio_signal, label), where audio_signal is a torch tensor representing
    the audio waveform and label is the corresponding label.

    Parameters:
    - dataloader (DataLoader): A DataLoader object that provides access to the
      audio data. The DataLoader should yield tuples of (audio_signal, label).

    Returns:
    - None: This function does not return any value. It is responsible for
      visualizing the audio data.
    """
    # Your visualization code goes here.
    # For example:
    for idx,batch in enumerate(dataloader):

        # Loading all the audio tensors
        waveforms = batch['waveform']
        mel_specs = batch['mel_spectrogram']
        augmented_mel_spec = batch['augmented_mel_spec']
        mfccs = batch['mfcc']
        augmented_waveforms = batch['augmented_waveform']
        augmented_log_mel_spectrogram = batch['augmented_log_mel_spec']
        
        random_sample = random.randint(1,waveforms.shape[0])-1
        # Perform visualization using just audio_signal
        # ...
        if vis_audio:
            audio_utils.AudioUtil.show_wave(aud=(waveforms[random_sample], SAMPLE_RATE),
                                label=f"Sample no: {random_sample} from batch: {idx}")
        
        # Perform visualization using before and after audio augmented
        """In PyTorch, tensors that require gradient calculation are used 
        for building computational graphs and performing automatic differentiation. 
        When you call tensor.numpy() on such a tensor, PyTorch raises this error 
        because the operation would break the computational graph and prevent gradient calculation.
        To resolve this issue, you need to detach the tensor from the computational 
        graph before converting it to a NumPy array. Here's how you can do it:
        The detach() method creates a new tensor that shares the same data as the original
        tensor but is detached from the computational graph. This means that any 
        changes made to the detached tensor will not affect the original tensor 
        or the computational graph. After detaching the tensor, you can safely call numpy() 
        to convert it to a NumPy array."""
        if vis_transfd_audio:
            audio_utils.AudioUtil.show_transform(orig=(waveforms[random_sample].detach(), SAMPLE_RATE),
                                     trans=(augmented_waveforms[random_sample].detach(),SAMPLE_RATE),
                                     label=f"Before and After transform sample no: {random_sample} from batch: {idx}")

        # Perform visualization using just mel spectrogram
        if vis_spectrogram:
            audio_utils.AudioUtil.show_spectro(spec=mel_specs[random_sample].detach(), 
                                   label=f"Mel spectrogram sample no: {random_sample} from batch: {idx}",
                                   figsize=(12,6))
        
        # Perform visualization using before and after mel spectrogram augmented
        if vis_transfd_spectrogram:
            audio_utils.AudioUtil.show_spectro(spec=augmented_mel_spec[random_sample].detach(), 
                                   label=f"Augmented Mel spectrogram sample no: {random_sample} from batch: {idx}",
                                   figsize=(6,6))
        
        # Perform visualization using just log mel spectrogram
        if vis_spectrogram:
            audio_utils.AudioUtil.show_spectro(spec=augmented_log_mel_spectrogram[random_sample].detach(), 
                                   label=f"Augmented Log Mel spectrogram sample no: {random_sample} from batch: {idx}",
                                   figsize=(6,6))
        
        # Perform visualization of MFCC
        if vis_mfcc:
            audio_utils.AudioUtil.show_spectro(spec=mfccs[random_sample].detach(), 
                                   label=f"MFCC sample no: {random_sample} from batch: {idx}",
                                   figsize=(6,6))
            
        break