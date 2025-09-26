FROM ghcr.io/jim60105/whisperx:no_model

WORKDIR /app

COPY server.py .
COPY requirements-docker.txt .

RUN python -m ensurepip

# Install dependencies required for the Flask server and WhisperX processing.
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

# Start the Flask server, listening on all interfaces on port 5000.
ENTRYPOINT ["/bin/sh", "-c", "flask --app server run --host=0.0.0.0 --port=5000"]