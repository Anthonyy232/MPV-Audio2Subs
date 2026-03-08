# MPV-Audio2Subs: Real-Time AI Local Subtitles for MPV

Generate high-quality, time-aligned subtitles for any video on the fly using local ASR with Qwen3-ASR.

## ✨ Features

- **100% Local**: All processing happens on your machine - no data sent externally
- **Real-Time Progress**: OSD feedback shows transcription progress
- **Smart Selection**: Respects your subtitle preferences (won't re-select if you turn it off)
- **Fast Startup**: Streaming audio extraction gets subtitles appearing quickly
- **GPU Accelerated**: Uses CUDA when available for fast transcription

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **FFmpeg** on PATH
- **MPV** with IPC enabled
- **NVIDIA GPU** (recommended) or CPU

### Installation

There are two ways to install MPV-Audio2Subs: globally (recommended for a clean MPV folder) or locally (by cloning the repo directly into MPV).

#### Global Installation (Recommended)
This installs the Python AI engine globally so you only need `main.lua` in your MPV plugins folder.

1. **Install the AI engine:**
```bash
# Using uv (Recommended - installs with optional ASR dependencies)
uv tool install git+https://github.com/Anthonyy232/MPV-Audio2Subs.git --with asr

# (Optional) If you have an RTX 50 series GPU and need CUDA 12.8, add the PyTorch index:
# uv tool install git+https://github.com/Anthonyy232/MPV-Audio2Subs.git --with asr --index-url https://download.pytorch.org/whl/cu128

# OR using pipx
pipx install git+https://github.com/Anthonyy232/MPV-Audio2Subs.git[asr]
```

2. **Install the MPV script:**
Download `main.lua` and place it in your MPV scripts folder (e.g., as `audio2subs.lua`).


### MPV Configuration

1. Enable IPC in `mpv.conf`:
```conf
input-ipc-server=/tmp/mpv-socket
```

2. Add keybinding in `input.conf`:
```conf
l script-message toggle_ai_subtitles
```

3. Ensure the script is in your MPV scripts folder:
   - **Windows**: `%APPDATA%\mpv\scripts\audio2subs.lua` (if global) or `%APPDATA%\mpv\scripts\audio2subs\` (if local)
   - **Linux/macOS**: `~/.config/mpv/scripts/audio2subs.lua` (if global) or `~/.config/mpv/scripts/audio2subs/` (if local)

### Usage

1. Start MPV with a video
2. Press `l` to activate AI subtitles
3. Wait for model to load (first time only)
4. Subtitles appear as transcription progresses!

## 📁 Project Structure

```
MPV-Audio2Subs/
├── main.lua              # MPV script with OSD and IPC
├── main.py               # Entry point (backward compatible)
├── src/audio2subs/       # Core Python package
│   ├── service.py        # Main service orchestrator
│   ├── engine.py         # Transcription engine
│   ├── mpv_client.py     # MPV IPC wrapper
│   ├── audio.py          # FFmpeg audio extraction
│   ├── subtitle.py       # ASS file generation
│   ├── config.py         # Configuration dataclasses
│   └── transcription/    # ASR backends
│       ├── base.py       # Interface
│       └── qwen.py       # Qwen3-ASR backend
├── pyproject.toml        # Modern packaging
└── requirements.txt
```

## ⚙️ Configuration

Environment variables:
- `AUDIO2SUBS_CHUNK_DURATION` - Chunk size in seconds (default: 300)
- `AUDIO2SUBS_PERSISTENT_MODE` - Keep model in memory (1/true)
- `AUDIO2SUBS_CPU_ONLY` - Force CPU mode (1/true)

## 🛠️ Troubleshooting

**Model Load Failures**
- Check `subtitle_service.log` for details
- Ensure dependencies are installed correctly (`pip install qwen-asr`)

**No Subtitles Appearing**
- Verify MPV IPC is configured correctly
- Check that FFmpeg can read the video

**Slow Performance**
- Ensure CUDA is available (`python -c "import torch; print(torch.cuda.is_available())"`)
- Increase chunk duration for longer batches

## 🔒 Privacy

All processing is 100% local. Audio never leaves your machine.