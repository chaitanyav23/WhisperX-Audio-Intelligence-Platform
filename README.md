# WhisperX Audio Transcription & Diarization

This project demonstrates an implementation of [WhisperX](https://github.com/m-bain/whisperX), providing fast automatic speech recognition (ASR) with word-level alignment and speaker diarization.

## Features

- **Fast Transcription**: Uses WhisperX (based on Faster-Whisper) for high-performance transcription.
- **Word-Level Alignment**: Precise word timestamps using phoneme-based alignment models.
- **Speaker Diarization**: Identification of different speakers in the audio using `pyannote-audio`.
- **Multi-language Support**: Demonstrated with English (`meeting.wav`) and Hindi (`news.wav`).

## Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support is highly recommended for performance.
- **Hugging Face Token**: Required to access the `pyannote/speaker-diarization-community-1` model.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install whisperx python-dotenv
   ```
   *Note: You may also need to install `ffmpeg` on your system.*

2. **Environment Variables**:
   Create a `.env` file in the root directory and add your Hugging Face token:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

3. **Audio Files**:
   Place your audio files (e.g., `.wav`) in the project directory.

## Usage

The main implementation is contained within the `whisperx_implementation.ipynb` Jupyter notebook. 

1. Open the notebook in your preferred environment (VS Code, JupyterLab, etc.).
2. Ensure your kernel is set to a Python environment where the dependencies are installed.
3. Run the cells sequentially to:
   - Load the Whisper model (`large-v2`).
   - Transcribe audio files.
   - Align the transcription for word-level timestamps.
   - Perform speaker diarization and assign speakers to transcript segments.

## File Structure

- `whisperx_implementation.ipynb`: The core notebook containing the implementation.
- `meeting.wav`: Sample audio file for English transcription and diarization.
- `news.wav`: Sample audio file for Hindi transcription.
- `.env`: (To be created) Configuration file for environment variables.
- `hf.txt`: Contains Hugging Face related info.

## Credits

- [WhisperX](https://github.com/m-bain/whisperX) by Max Bain.
- [OpenAI Whisper](https://github.com/openai/whisper).
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio).
