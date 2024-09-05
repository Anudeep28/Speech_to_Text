import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import audio_utils

class WhisperDataset(Dataset):
    def __init__(self, csv_file: pd.DataFrame, 
                 root_dir: str, 
                 target_length: int=16000*30, 
                 expected_time_step: int=3000,
                 signal_shift: bool=False,
                 time_mask_param: int=5,
                 freq_mask_param: int=5,
                 augment_both: bool=False,
                 SAMPLE_RATE: int=16000,
                 N_FFT: int=400,
                 N_MELS: int=80,
                 N_MFCC: int=40):
        self.data = csv_file
        self.root_dir = root_dir
        self.target_length = target_length
        self.expected_time_step = expected_time_step
        self.signal_shift = signal_shift
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.aug_both = augment_both
        self.SAMPLE_RATE = SAMPLE_RATE
        self.N_FFT = N_FFT
        self.N_MELS = N_MELS
        self.N_MFCC = N_MFCC

        # Pre-compute transforms
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
        self.amptodeb = T.AmplitudeToDB(top_db=N_MELS)
        self.resampler = T.Resample(48000, SAMPLE_RATE)  # Assuming 48kHz is the most common original sample rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        audio_path = item['audio']
        transcript = item['text']

        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != self.SAMPLE_RATE:
            waveform = self.resampler(waveform)

        waveform = self._prepare_waveform(waveform)
        mfcc = self.mfcc(waveform)

        augmented_waveform = self.apply_augmentation_waveform(waveform)
        mel_spec, augmented_log_mel_spec = self._compute_mel_spectrograms(waveform, augmented_waveform)

        return {
            'waveform': waveform,
            'mel_spectrogram': mel_spec,
            #'augmented_mel_spec': augmented_log_mel_spec,
            'mfcc': mfcc,
            'augmented_waveform': augmented_waveform,
            'augmented_log_mel_spec': augmented_log_mel_spec,
            'transcript': transcript
        }

    def _prepare_waveform(self, waveform):
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim or pad waveform to target length
        waveform_length = waveform.shape[-1]
        if waveform_length > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform_length < self.target_length:
            padding = self.target_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
        
        return waveform

    def _compute_mel_spectrograms(self, waveform, augmented_waveform):
        
        if self.aug_both:
            # compute mel spec <- augmented waveform
            mel_spec = self.mel_spectrogram(augmented_waveform)
            # compute log mel spec <- mel_spec <- augmented
            log_mel_spec = torch.log(mel_spec + 1e-9)
            # # compute aug log mel <- leg_mel <- mel_spec <- augmented waveform
            # augmented_mel_spec = self.mel_spectrogram(augmented_waveform)
        else:
            # compute mel spectrogram from non augmented waveform & augmented
            mel_spec = self.mel_spectrogram(waveform)
            # compute log mel spec <- mel_spec <- non augmented
            log_mel_spec = torch.log(mel_spec + 1e-9)
            # # compute aug log mel <- leg_mel <- mel_spec <- augmented waveform
            # augmented_mel_spec = self.mel_spectrogram(augmented_waveform)

        augmented_log_mel_spec = self.apply_augmentation_mel(log_mel_spec)
        
        augmented_log_mel_spec = self._pad_or_trim_spectrogram(augmented_log_mel_spec)
        
        return mel_spec, augmented_log_mel_spec

    def _pad_or_trim_spectrogram(self, spectrogram):
        if spectrogram.shape[-1] > self.expected_time_step:
            return spectrogram[:, :, :self.expected_time_step]
        elif spectrogram.shape[-1] < self.expected_time_step:
            padding = self.expected_time_step - spectrogram.shape[-1]
            return torch.nn.functional.pad(spectrogram, (0, padding), mode='constant', value=0)
        return spectrogram

    def apply_augmentation_waveform(self, waveform):
        if self.signal_shift:
            waveform = audio_utils.AudioUtil.signal_shift((waveform, self.SAMPLE_RATE), max_shift_pct=0.30)[0]
            
        n_steps = int(torch.randint(-4, 5, (1,)).item())
        pitch_shift = T.PitchShift(self.SAMPLE_RATE, n_steps=n_steps)
        waveform = pitch_shift(waveform)
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise

    def apply_augmentation_mel(self, log_mel_spec):
        return audio_utils.AudioUtil.spectro_augment(log_mel_spec)

def collate_fn(batch):
    keys = ['waveform', 'mel_spectrogram', 'mfcc', 'augmented_waveform', 'augmented_log_mel_spec']
    padded_batch = {key: torch.nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True, padding_value=0.0) for key in keys}
    padded_batch['transcript'] = [item['transcript'] for item in batch]
    return padded_batch

# def create_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
