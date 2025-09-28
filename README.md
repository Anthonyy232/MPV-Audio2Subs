# MPV-Audio2Subs: Real-Time AI Local Subtitles for MPV

A small, self-contained local service that generates high-quality, time-aligned subtitles for your videos on the fly using local Automatic Speech Recognition (ASR) models.

While a video plays in MPV, the service extracts audio, transcribes it locally with a pluggable ASR backend, and writes atomic, time-aligned ASS subtitle files right next to the video.

## ✨ Features

This service is designed for responsiveness and maximum local control:

*   **100% Local Execution:** Audio extraction (via FFmpeg) and transcription occur entirely on your local machine. No data is sent to external services by default.
*   **On-Demand Subtitles:** Transcribe any video in your library automatically after pressing a single hotkey in MPV.
*   **Intelligent Chunking:** Uses chunked transcription with priority given to audio segments near the current playback position, significantly improving responsiveness after seeks.
*   **Pluggable ASR Backend:** Default client uses **NVIDIA NeMo's ASR** (Parakeet models) for fast, high-quality transcription, especially when leveraging a CUDA GPU. Easily switch to other local or remote backends.
*   **Atomic Updates:** Subtitles are saved as `<video_basename>.ai.ass` and updated atomically to ensure MPV always loads a complete, valid subtitle file.
*   **IPC Communication:** Seamless integration with MPV via its IPC socket for audio requests, control signals, and subtitle file reload notifications.

## 🚀 Installation (Quick Start)

This project requires MPV, FFmpeg, and Python 3.9+. We strongly recommend using the provided installation scripts to handle Python dependencies within a virtual environment (`venv`).

### 1. Prerequisites

1.  **MPV:** Must be installed and configured with an **IPC server enabled** (see Configuration below).
2.  **FFmpeg:** Must be available on your system `PATH` (used for fast audio extraction).
3.  **Python 3.9+:** Required for the core transcription service.

### 2. Copy Files

Copy the entire `audio2subs` folder into MPV's user scripts directory:

*   **Linux/macOS:** `~/.config/mpv/scripts/`
*   **Windows:** `%APPDATA%\mpv\scripts\`

### 3. Run Setup Script

Navigate into the copied `audio2subs` directory and run the appropriate setup script. These scripts create a local Python `venv`, install dependencies from `requirements.txt`.

```bash
# Linux / macOS
./install.sh
```

```powershell
# Windows (PowerShell / cmd)
install.bat
```

> **Note on CUDA:** The installation scripts attempt to install a PyTorch wheel targeting a common CUDA version for immediate GPU acceleration. If you encounter issues, you may need to manually install PyTorch compatible with your specific hardware and drivers *before* running the script.

## ⚙️ MPV Configuration

The Python service relies on MPV's IPC socket for communication.

### 1. Enable IPC Server

Add the following line to your `mpv.conf`:

```conf
# Linux/macOS Example
input-ipc-server=/tmp/mpv-socket
# Windows Example
input-ipc-server=\\.\pipe\mpv-socket
```

### 2. Set Input Binding

Bind a key in your `input.conf` to toggle the service. This key triggers the Lua script, which manages starting the Python client if it isn't already running.

```conf
# Example: Press 'l' (letter L) to toggle the service
l script-message toggle_ai_subtitles
```

## 🎬 Usage

1.  Start MPV with IPC configured.
2.  Load a video file.
3.  Press the configured toggle key (e.g., `l`).

The Lua script will attempt to start `main.py` from the virtual environment. The service will then:

1.  Extract the audio from the video using FFmpeg.
2.  Start background transcription.
3.  As chunks are transcribed, the `.ai.ass` file is created/updated.
4.  MPV is notified to automatically load or reload the subtitle file.

## 🛠️ Repository Layout

| File / Directory | Description |
| :--- | :--- |
| `main.lua` | The MPV-side script. Handles the toggle hotkey, launches the Python client if necessary, and forwards playback events (seeks, loads). |
| `main.py` | The main Python service handler. Connects to MPV via IPC, manages the lifecycle of per-video transcription engines. |
| `ai_engine.py` | The core, per-video processing engine. Manages FFmpeg audio extraction, schedules chunk transcription, groups words, and writes atomic ASS files. |
| `transcription.py` | Defines the `TranscriptionInterface` and the default `ParakeetLocalClient` (using NeMo). |
| `install.bat` / `install.sh` | Helpers for setting up the Python environment and dependencies. |
| `requirements.txt` | Python dependencies. |

## 🧩 Configuration & Extensibility

### Chunk Size

The balance between responsiveness and model efficiency is controlled by the chunk duration:

*   **`CONFIG['CHUNK_DURATION_SECONDS']`** in `main.py` (Default: 30s).

### Pluggable Transcription Backends

The service is designed to allow easy swapping of ASR providers:

1.  Implement a new class that inherits from `TranscriptionInterface` in `transcription.py`.
2.  Ensure your implementation returns a standard `words` structure: a list of objects containing `word`, `start` (timestamp in seconds), and `end` (timestamp in seconds).
3.  Update the configuration in `main.py` to use your custom client class.

## 🗃️ Logs and Artifacts

*   **`<video_basename>.ai.ass`:** The generated subtitle file, saved next to the source video.
*   **`subtitle_service.log`:** Created in the script directory, containing startup messages, model load diagnostics, and FFmpeg errors.
*   **Lock File:** The service creates a lock file in the system temp directory (`tempfile.gettempdir()`) to prevent multiple instances of the service from running concurrently.

## 🛑 Troubleshooting

**1. Model Load Failures (ASR)**

*   **Check the Log:** Review `subtitle_service.log` for Python tracebacks.
*   **Dependencies:** Ensure `nemo_toolkit[asr]` is installed (required for the default client).
*   **GPU Issues:** If using CUDA, verify that your installed PyTorch version is compatible with your GPU drivers and that the model is correctly moving to the device.

**2. MPV IPC Connection Errors**

*   Ensure the `input-ipc-server` path in your `mpv.conf` exactly matches the path the Python client is trying to connect to (usually `/tmp/mpv-socket` or `\\.\pipe\mpv-socket`).
*   Verify that no other application is using that IPC path.

**3. FFmpeg Errors**

*   Confirm that `ffmpeg` is available on your system `PATH`.
*   Ensure the service has read access to the video file (FFmpeg needs to read the video to extract the PCM audio).

**4. Stale Lock File**

If the service crashed unexpectedly, a stale lock file may prevent new instances from launching. If you confirm no service is running, manually delete the lock file from your OS temporary folder.

## 🔒 Security & Privacy

This project is built to run entirely locally by default. Audio processing and transcription are performed on the same machine. No data is ever transmitted beyond your machine.