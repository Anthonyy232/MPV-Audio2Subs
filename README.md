# MPV-Audio2Subs

This project provides a high-quality, real-time transcription subtitle generation service for the MPV media player, leveraging WhisperX for transcription and alignment. It is designed to generate professionally timed subtitles for any video file as you watch it.

Languages supported: English (en)

## 🚀 Features

*   **Real-Time Transcription:** Subtitles are generated chunk-by-chunk in the background while the video plays.
*   **Seek Awareness:** The system prioritizes transcription tasks based on the current playback position and automatically clears the queue upon seeking.
*   **Professional Refinement:** Subtitles are post-processed to adhere to professional standards (e.g., character-per-second limits, minimum/maximum duration, pause-based line splitting).
*   **Persistent AI Backend:** Uses a persistent Docker container to keep the large AI models loaded in memory, minimizing startup time after the initial launch.

## 🏗️ Architecture and Stack

The system operates as a three-part distributed application communicating via IPC and network sockets:

| Component | Technology | Role | Communication |
| :--- | :--- | :--- | :--- |
| **1. MPV Player** | C/C++, Lua | Frontend, Video Playback | IPC (JSON-IPC) |
| **2. Service Manager** | Lua (`main.lua`) | Lifecycle Management, Docker Control | Shell/Docker CLI |
| **3. Python Client** | Python (`main.py`) | Task Orchestration, MPV Integration | IPC (MPV Socket), HTTP (Flask) |
| **4. AI Server** | Python, Flask, Docker, WhisperX | Heavy Lifting (Transcription, Alignment) | HTTP (localhost:5000) |

### Key Libraries and Tools

*   **AI/ML:** `WhisperX` (for transcription and word-level alignment).
*   **Containerization:** `Docker` (for GPU isolation and environment consistency).
*   **Media Processing:** `FFMPEG` (used by the Python client to extract raw audio chunks).
*   **IPC:** `python-mpv-jsonipc` (for Python client communication with MPV).

## Configuration and Performance Tuning

If you find the application's transcription speed too slow or resource usage too high, you can switch to a smaller Whisper model. The default model used is `large-v3`.

Available model options (e.g., `base`, `small`, `medium`) are listed in the dockerized WhisperX documentation:
[https://github.com/jim60105/whisperX](https://github.com/jim60105/whisperX)

To change the model, you must modify the default value in the following two files:

1.  **`main.lua`**: Update the value associated with `-e` and the environment variable definition:
    ```lua
    WHISPER_MODEL=large-v3
    ```

2.  **`server.py`**: Update the default fallback value in the model size definition:
    ```python
    MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
    ```

## 📋 Prerequisites

Before installation, ensure you have the following installed and configured:

1.  **MPV Media Player:** Must be installed.
2.  **Docker Desktop/Daemon:** Required to run the WhisperX AI server container. Must be running before starting the service.
3.  **FFMPEG:** Must be installed and accessible in your system's PATH.
4.  **Python 3.9+**

### Docker (Windows, Linux, macOS)

This project requires Docker on all platforms (Windows, Linux, and macOS) to run the WhisperX AI server container. Installation steps vary by OS; please consult the official Docker documentation or search for platform-specific guides online.

Quick verification (after installing Docker and starting the Docker daemon):

```bash
# Check Docker is available
docker version

# Run a quick test container
docker run --rm hello-world
```

If you are on Windows you may need Docker Desktop with WSL2 integration for the best compatibility with Linux-based images. If you need platform-specific instructions, look for the official Docker installation guide for your OS.

## ⚙️ Installation and Setup (Step-by-Step)

This setup requires placing the project files into a specific directory structure recognized by MPV, followed by running the installation script and configuring MPV's IPC.

### Step 1: Place the Project Files

You must place the entire project folder, named `ai_service_manager`, into your MPV scripts directory.

| Operating System | Default MPV Scripts Location |
| :--- | :--- |
| **Linux/macOS** | `~/.config/mpv/scripts/` |
| **Windows** | `%APPDATA%\mpv\scripts\` or `[MPV Install Dir]\scripts\` |

The final directory structure should look like this (note that `mpv_service_manager.lua` must be renamed to `main.lua`):

```
[MPV Scripts Folder]/
└── ai_service_manager/
    ├── ai_engine.py
    ├── Dockerfile
    ├── install.bat
    ├── install.sh
    ├── main.lua          <-- MPV's entry point for the Lua script
    ├── main.py           <-- The Python client service
    ├── README.md
    ├── requirements-docker.txt
    ├── requirements.txt
    ├── server.py
    └── transcription.py
```

### Step 2: Run the Installation Script

Navigate into the `ai_service_manager` directory in your terminal and run the appropriate script to set up the Python virtual environment (`venv`) and install dependencies.

**For Windows Users:**

```bash
install.bat
```

**For Linux/macOS Users:**

```bash
chmod +x install.sh
./install.sh
```

### Step 3: Configure MPV IPC Socket (`mpv.conf`)

The Python client needs to communicate with MPV via an Inter-Process Communication (IPC) socket. You must enable the IPC server in your `mpv.conf` file.

Add **one** of the following lines to your `mpv.conf` file, depending on your operating system:

| Operating System | `mpv.conf` Setting |
| :--- | :--- |
| **Linux/macOS** | `input-ipc-server=/tmp/mpv-socket` |
| **Windows** | `input-ipc-server=\\.\pipe\mpv-socket` |

### Step 4: Bind a Key for Toggling (`input.conf`)

Add a key binding to your `input.conf` file to easily start and stop the service using the Lua script:

```conf
# Binds the 'l' key to toggle the AI subtitle service
l script-message toggle_ai_subtitles
```

## 🚀 Usage and Service Lifecycle

### Initial Startup (First Run Only)

The very first time you run the service, the Docker image must be built. This process downloads the large AI model and can take **5 to 15 minutes** depending on your internet speed and CPU/GPU.

1.  **Start Docker Desktop/Daemon.**
2.  **Start MPV** and open a video.
3.  **Press the bound key (e.g., `l`).**
    *   MPV will display: `AI Subtitles: Building Docker image...`
    *   Wait for the build to complete. MPV will then display: `AI Service: Starting... (Loading model, please wait)`
    *   The service is ready when the OSD confirms: `AI Subtitle Service: ON`.

### Subsequent Usage

After the initial build, the Docker image is cached, and the container is persistent.

1.  **Start Docker Desktop/Daemon.**
2.  **Start MPV** and open a video.
3.  **Press the bound key (e.g., `l`).**
    *   The container starts quickly, but the AI model still needs to load onto the GPU/CPU. Wait for the `(Loading model, please wait)` message to clear.
4.  **Real-Time Transcription:** As the video plays, the Python client queues chunks for transcription. The generated subtitles will appear in real-time and update the `.ai.ass` file next to your video.

### Stopping the Service

Press the bound key again (e.g., `l`). The Lua script will send a shutdown command to the Python client and forcefully stop and remove the persistent Docker container, freeing up GPU resources.

## 🐛 Troubleshooting and Logging

The Python client generates a detailed log file that is essential for debugging connection issues or transcription failures:

*   **Log File Location:** `[script_directory]/subtitle_service.log`
    *(Note: This file is overwritten every time the Python client starts.)*

If the service fails to start:

1.  **Check Docker Status:** Ensure Docker is running and that the container `mpv_whisperx_instance` is not stuck in a failed state.
2.  **Check Logs:** Review `subtitle_service.log` for `[FATAL]` or `[CRITICAL]` errors.
    *   If you see errors connecting to `http://localhost:5000`, the Docker container failed to start or the model failed to load.
    *   If you see errors connecting to the MPV socket, double-check your `mpv.conf` setting matches the path expected by your OS.
