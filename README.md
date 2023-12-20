# EngVision Whisper Service

## Overview

EngVision Whisper is a speech-to-text service leveraging the Whisper model from OpenAI. It's designed to transcribe audio files efficiently and accurately. The service is built using Flask and can be easily deployed in a containerized environment.

## Requirements

- Docker
- Python 3.7 or later

## Installation and Setup

### Building the Docker Image

1. **Clone the Repository**: Clone the repository to your local machine.
   ```
   git clone https://github.com/EngVision/Whisper-Service
   cd Whisper-Service
   ```
2. **Build Docker Image**:
   ```
   docker build -t engvision-whisper .
   ```

### Running the Container

Run the Docker container:

```
docker run -p 8000:8000 engvision-whisper
```

## Usage

### Starting the Service

Once the container is running, the service will be available at `http://localhost:8000/`.

### API Endpoints

1. **Root Endpoint** (`GET /`):

   - Description: A simple endpoint to confirm that the service is running.
   - Response: "Hello from EngVision Whisper!"

2. **Speech to Text Endpoint** (`POST /stt`):

   - Description: Upload an audio file for transcription.
   - Request: A `multipart/form-data` request with the audio file.
   - Response: A JSON containing the `status` and `check_url` for the transcription task.

3. **Check Status Endpoint** (`GET /stt/<file_id>`):
   - Description: Check the status of a transcription task.
   - Response: A JSON with the `status` of the task and the `result` if completed.

### Example Usage

1. **Uploading an Audio File for Transcription**:
   ```
   curl -F "file=@path_to_audio_file" http://localhost:8000/stt
   ```
2. **Checking Transcription Status**:
   ```
   curl http://localhost:8000/stt/<file_id>
   ```

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your changes.
