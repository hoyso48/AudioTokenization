import torch
import torchaudio
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi

def stft(x, fft_size, hop_size, win_length, window, use_complex=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """

    x_stft = torch.stft(x, fft_size, hop_size, win_length, window.to(x.device),
                        return_complex=True)

    # clamp is needed to avoid nan or inf
    if not use_complex:
        return torch.sqrt(torch.clamp(
            x_stft.real ** 2 + x_stft.imag ** 2, min=1e-7, max=1e3)).transpose(2, 1)
    else:
        res = torch.cat([x_stft.real.unsqueeze(1), x_stft.imag.unsqueeze(1)], dim=1)
        res = res.transpose(2, 3) # [B, 2, T, F]
        return res


class KaldiMelSpectrogram(nn.Module):
    def __init__(self, n_mels=128, sr=16000, win_length=800, hopsize=320, n_fft=1024,
                 htk=False, fmin=0.0, fmax=None, norm=1):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr / 2
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize

        # Hann window buffer for STFT
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)

        # Pre-emphasis filter coefficient buffer
        self.register_buffer("preemphasis_coefficient", 
                             torch.as_tensor([[[-.97, 1]]]), 
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input audio waveform to Kaldi-style Mel Spectrogram.
        
        Args:
            x (torch.Tensor): Input waveform. shape: (batch, time)
        
        Returns:
            torch.Tensor: Mel Spectrogram.
        """
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
                       
        x = (x ** 2).sum(dim=-1)
        
        mel_basis, _ = kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                           self.fmin, self.fmax, vtln_low=100.0, 
                                           vtln_high=-500., vtln_warp_factor=1.0)
        
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x.float())

        melspec = (melspec + 0.00001).log()

        melspec = (melspec + 4.5) / 5.0

        return melspec