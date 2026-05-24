import whisperx
import torch
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "int8"

import librosa
import whisperx
import torch

from whisperx.diarize import DiarizationPipeline

def load_diarization_model():

    print("Loading diarization model...")

    diarize_model = DiarizationPipeline(
        token=HF_TOKEN,
        device=DEVICE
    )

    return diarize_model

def validate_audio_length(audio_path, max_duration=120):
    """
    Limit audio duration for HF Spaces.
    """

    duration = librosa.get_duration(path=audio_path)

    if duration > max_duration:
        raise ValueError(
            f"Audio too long. "
            f"Maximum allowed length is "
            f"{max_duration} seconds."
        )

def load_models():
    """
    Load WhisperX ASR model.
    """

    print(f"Using device: {DEVICE}")

    asr_model = whisperx.load_model(
        "small",
        DEVICE,
        compute_type=COMPUTE_TYPE
    )

    return asr_model

def format_transcript(segments):
    """
    Convert segments into readable speaker transcript.
    """

    formatted = []

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start = round(seg["start"], 2)
        end = round(seg["end"], 2)
        text = seg["text"]

        formatted.append(
            f"[{start}s - {end}s] {speaker}: {text}"
        )

    return "\n".join(formatted)
