from dataclasses import dataclass, field
from typing import List
import torch
import torchaudio

@dataclass
class CNNFeaturesSetting:
    frontend_algorithm: List[str] = field(default_factory=lambda: ["mfcc"])
    use_spectrogram: bool = True
    use_deltas: bool = True  # Added: Highly effective for catching fake voice transitions

SAMPLING_RATE = 16_000
win_length = 400  
hop_length = 160  

device = "cuda" if torch.cuda.is_available() else "cpu"

# FIX 1: Explicitly pass "n_mels": 80 inside melkwargs to eliminate empty filterbanks
MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=80,
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
        "n_mels": 80,  # Explicitly match your target dimension
    },
).to(device)

LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=80,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=80,
    n_stft=257,
    sample_rate=SAMPLING_RATE,
).to(device)


def prepare_feature_vector(
    audio: torch.Tensor,
    cnn_features_setting: CNNFeaturesSetting,
    win_length: int = 400,
    hop_length: int = 160,
) -> torch.Tensor:

    feature_vector = []

    # Extract base features
    if "mfcc" in cnn_features_setting.frontend_algorithm:
        mfcc_feature = MFCC_FN(audio)
        feature_vector.append(mfcc_feature)
        if cnn_features_setting.use_deltas:
            # Catch dynamic acceleration artifacts
            feature_vector.append(torchaudio.functional.compute_deltas(mfcc_feature))

    if "lfcc" in cnn_features_setting.frontend_algorithm:
        lfcc_feature = LFCC_FN(audio)
        feature_vector.append(lfcc_feature)
        if cnn_features_setting.use_deltas:
            feature_vector.append(torchaudio.functional.compute_deltas(lfcc_feature))

    if cnn_features_setting.use_spectrogram:
        stft_abs_mel, stft_abs_angle = prepare_stft_features(audio, win_length, hop_length)
        feature_vector.append(stft_abs_mel)
        feature_vector.append(stft_abs_angle)

    assert len(feature_vector) >= 1, "Feature vector must contain at least one feature!"

    # Ensure everything is neatly aligned and stack along the channel dimension
    # Format: [batch_size, feature_num, 80, frames]
    feature_vector = torch.stack(feature_vector, dim=1)
    return feature_vector


def prepare_stft_features(audio, win_length, hop_length):
    stft_out = torch.stft(
        audio,
        n_fft=512,
        return_complex=True,
        hop_length=hop_length,
        win_length=win_length,
    )

    # FIX 2: Compute magnitude and phase directly on the true complex tensor FIRST
    magnitude = torch.abs(stft_out)
    phase = torch.angle(stft_out)

    # Apply Mel scale cleanly to the clean magnitude values
    stft_abs_mel = MEL_SCALE_FN(magnitude)

    # Downsample phase linearly to match the (80, frames) shape requirement 
    # instead of passing it through a Mel filter matrix which destroys it
    stft_abs_angle = torch.nn.functional.interpolate(
        phase.unsqueeze(1), 
        size=(80, phase.shape[-1]), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)

    return stft_abs_mel, stft_abs_angle