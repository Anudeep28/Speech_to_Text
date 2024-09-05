"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
import audio_utils


# Try to use whisper mel and resampler

import torchaudio
import torchaudio.transforms as T

class WhisperDataset(Dataset):
    def __init__(self, csv_file: pd.DataFrame, 
                 root_dir: str, 
                 target_length: int=16000*30, 
                 expected_time_step: int=3000,
                 time_mask_param: int=5,
                 freq_mask_param: int=5,
                 augment_both: bool=False,
                 SAMPLE_RATE: int=16000,
                 N_FFT: int=400,
                 N_MELS: int=80,
                 N_MFCC: int=40):
        self.data = csv_file #pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_length = target_length
        self.expected_time_step = expected_time_step
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.aug_both = augment_both
        self.SAMPLE_RATE = SAMPLE_RATE
        self.N_FFT = N_FFT
        self.N_MELS = N_MELS
        self.N_MFCC = N_MFCC
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS
        )
        
        self.mfcc = T.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={'n_fft': N_FFT, 'n_mels': N_MELS}
        )
        # Amplitude to decibels
        self.amptodeb = T.AmplitudeToDB(top_db=N_MELS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        audio_path = item['audio']
        transcript = item['text']

        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != self.SAMPLE_RATE:
            resampler = T.Resample(sample_rate, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Trim or pad waveform to target length
        waveform_length = waveform.shape[-1]
        # print(f"waveform length: {waveform_length}")
        if waveform_length > self.target_length:
            waveform = waveform[:, :self.target_length]  # Trim
        elif waveform_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        

        # Compute MFCC ( Later have to look how to use it in Whisper model for more accuracy)
        mfcc = self.mfcc(waveform)

        # Audio augmentation
        augmented_waveform = self.apply_augmentation_waveform(waveform, sample_rate=sample_rate)

        # Compute mel spectrogram depending on waveform augmentation
        if self.aug_both:
            # 1. Compute mel spectrogram from augmented waveform
            mel_spec = self.mel_spectrogram(augmented_waveform)
            # 2. Compute log of mel spectrogram (This we have two different ways of computing the log)
            log_mel_spec = torch.log(mel_spec + 1e-9)
            # 3. Augment the log mel spectrogram
            augmented_log_mel_spec = self.apply_augmentation_mel(log_mel_spec)
            # DO I need to add Decibel scale here?
            augmented_mel_spec = self.apply_augmentation_mel(mel_spec)
            # ----------------Logarithmic way ----------------- #
            # augmented_log_mel_spec = torch.log(augmented_mel_spec + 1e-9)
            # ----------------Amplitude to Decibels ----------------- #
            # augmented_log_mel_spec = self.amptodeb(augmented_mel_spec)
        else:
            # 1. Compute mel spectrogram
            mel_spec = self.mel_spectrogram(waveform)
            # 2. Compute log of mel spectrogram (This we have two different ways of computing the log)
            log_mel_spec = torch.log(mel_spec + 1e-9)
            # 3. Augment the log mel
            augmented_log_mel_spec = self.apply_augmentation_mel(log_mel_spec)
            # Augmented mel spectrogram
            augmented_mel_spec = self.apply_augmentation_mel(mel_spec)
            # ----------------Logarithmic way ----------------- #
            # augmented_log_mel_spec = torch.log(augmented_mel_spec + 1e-9)
            # ----------------Amplitude to Decibels ----------------- #
            # augmented_log_mel_spec = self.amptodeb(augmented_mel_spec)

        # Pad or trim the log-mel spectrogram to sequence length to 1600 for the whisper model
        # expected_time_step = 3000  # Set this based on your requirements
        if augmented_log_mel_spec.shape[-1] > self.expected_time_step:
            augmented_log_mel_spec = augmented_log_mel_spec[:, :, :self.expected_time_step]  # Trim
        elif augmented_log_mel_spec.shape[-1] < self.expected_time_step:
            padding = self.expected_time_step - augmented_log_mel_spec.shape[-1]
            augmented_log_mel_spec = torch.nn.functional.pad(augmented_log_mel_spec, (0, padding), mode='constant', value=0)  # Pad
        
        # print(f"After padding and trimming log_mel_spec ahape: {augmented_log_mel_spec.shape}")

        return {
            'waveform': waveform,
            'mel_spectrogram': mel_spec,
            'augmented_mel_spec':augmented_mel_spec,
            'mfcc': mfcc,
            'augmented_waveform': augmented_waveform,
            'augmented_log_mel_spec': augmented_log_mel_spec,
            'transcript': transcript
        }

    def apply_augmentation_waveform(self, waveform, sample_rate):
        """
        This method applies 3 augmentations to the waveform
        1. Signal shift augmentation
        2. Pitch shift augmentation
        3. Gaussian noise augmentation

        Note: Further options would be provided to the User
        """
        # Check if the waveform is long enough for time stretching
        # if waveform.shape[-1] > 2000:  # Arbitrary threshold, adjust as needed
        #     stretch = random.uniform(0.8, 1.2)
        #     time_stretch = T.TimeStretch()
        #     waveform = time_stretch(waveform.unsqueeze(0), stretch).squeeze(0)
        
        # ----------------------------
        # Shifts the signal to the left or right by some percent. Values at the end
        # are 'wrapped around' to the start of the transformed signal.
        # ----------------------------
        # Thinking on removing signal shift
        #waveform = audio_utils.AudioUtil.signal_shift((waveform,sample_rate),max_shift_pct=0.30)
        # Pitch shift
        n_steps = int(random.uniform(-4, 4))
        pitch_shift = T.PitchShift(self.SAMPLE_RATE,n_steps=n_steps)
        waveform = pitch_shift(waveform)

        # Add Gaussian noise
        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

        return waveform
    
    def apply_augmentation_mel(self, log_mel_spec):
        """
        # ----------------------------
        # Augment the Spectrogram by masking out some sections of it in both the frequency
        # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
        # overfitting and to help the model generalise better. The masked sections are
        # replaced with the mean value.
        # ----------------------------

        Note: Have to change the variable name to avoid confusion

        """
        ##################### Using Spectro Augment from audio utils ###########

        log_mel_spec = audio_utils.AudioUtil.spectro_augment(log_mel_spec)

        #######################################################################
        # Old Way of Masking Time and Frequency
        # that is why commenting but can be used when required
        # Time masking
        # time_mask = T.TimeMasking(time_mask_param=self.time_mask_param)
        # log_mel_spec = time_mask(log_mel_spec)

        # # Frequency masking
        # freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
        # log_mel_spec = freq_mask(log_mel_spec)

        return log_mel_spec

def collate_fn(batch):
    waveforms = [item['waveform'] for item in batch]
    mel_specs = [item['mel_spectrogram'] for item in batch]
    augmented_mel_spec = [item['augmented_mel_spec'] for item in batch]
    mfccs = [item['mfcc'] for item in batch]
    augmented_waveforms = [item['augmented_waveform'] for item in batch]
    augmented_log_mel_specs = [item['augmented_log_mel_spec'] for item in batch]
    transcripts = [item['transcript'] for item in batch]

    # Pad waveforms and augmented waveforms to the same length
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    augmented_waveforms = torch.nn.utils.rnn.pad_sequence(augmented_waveforms, batch_first=True, padding_value=0.0)

    # Pad mel spectrograms and MFCCs
    mel_specs = torch.nn.utils.rnn.pad_sequence(mel_specs, batch_first=True, padding_value=0.0)
    augmented_mel_spec = torch.nn.utils.rnn.pad_sequence(augmented_mel_spec, batch_first=True, padding_value=0.0)
    mfccs = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True, padding_value=0.0)
    augmented_log_mel_specs = torch.nn.utils.rnn.pad_sequence(augmented_log_mel_specs, batch_first=True, padding_value=0.0)

    return {
        'waveform': waveforms,
        'mel_spectrogram': mel_specs,
        'augmented_mel_spec':augmented_mel_spec,
        'mfcc': mfccs,
        'augmented_waveform': augmented_waveforms,
        'augmented_log_mel_spec': augmented_log_mel_specs,
        'transcript': transcripts
    }

