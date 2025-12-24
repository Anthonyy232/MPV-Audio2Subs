# MPV-Audio2Subs: Real-Time AI Local Subtitles for MPV

Generate high-quality, time-aligned subtitles for any video on the fly using local ASR with NVIDIA Parakeet.

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

```bash
# Clone and install
cd MPV-Audio2Subs
./install.sh    # Linux/macOS
install.bat     # Windows
```

### MPV Configuration

1. Enable IPC in `mpv.conf`:
```conf
input-ipc-server=/tmp/mpv-socket
```

2. Add keybinding in `input.conf`:
```conf
l script-message toggle_ai_subtitles
```

3. Copy the folder to MPV scripts:
   - **Linux/macOS**: `~/.config/mpv/scripts/audio2subs/`
   - **Windows**: `%APPDATA%\mpv\scripts\audio2subs\`

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
│       └── parakeet.py   # NeMo Parakeet
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
- Ensure NeMo is installed: `pip install nemo_toolkit[asr]`

**No Subtitles Appearing**
- Verify MPV IPC is configured correctly
- Check that FFmpeg can read the video

**Slow Performance**
- Ensure CUDA is available (`python -c "import torch; print(torch.cuda.is_available())"`)
- Increase chunk duration for longer batches

## 🔒 Privacy

All processing is 100% local. Audio never leaves your machine.