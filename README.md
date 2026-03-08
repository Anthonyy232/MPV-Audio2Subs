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
uv tool install "git+https://github.com/Anthonyy232/MPV-Audio2Subs.git#egg=audio2subs[asr]" --extra-index-url https://download.pytorch.org/whl/cu128

# OR using pipx
pipx install "git+https://github.com/Anthonyy232/MPV-Audio2Subs.git#egg=audio2subs[asr]"
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

## ⚡ Flash Attention 2 (Optional, Recommended)

Flash Attention 2 speeds up transcription, especially for longer audio. Because it requires a prebuilt binary matching your exact Python, PyTorch, and CUDA versions, it must be installed manually.

**Step 1 — Check your versions:**
```bash
python -c "import sys, torch; print(f'Python: {sys.version.split()[0]}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

Example output:
```
Python: 3.11.9
PyTorch: 2.10.0+cu128
CUDA: 12.8
```

**Step 2 — Find and install the matching wheel:**

Go to [mjunya.com/flash-attention-prebuild-wheels](https://mjunya.com/flash-attention-prebuild-wheels/) and download the wheel matching your `cpXYZ` (Python), `cuXYZ` (CUDA), and `torchX.Y` versions.

For example, Python 3.11 + CUDA 12.8 + PyTorch 2.10:
```bash
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-win_amd64.whl"
```

Once installed, the service auto-detects it — you'll see `Flash Attention 2 enabled` in the log on next startup.

## ⚙️ Configuration

Environment variables:
- `AUDIO2SUBS_CHUNK_DURATION` - Chunk size in seconds (default: 30)
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
- Verify CUDA is available and your versions with the command above
- Install Flash Attention 2 (see above)
- Check `subtitle_service.log` for `Flash Attention 2 enabled` to confirm it's active

## 🔒 Privacy

All processing is 100% local. Audio never leaves your machine.