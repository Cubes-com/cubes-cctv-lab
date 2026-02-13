# AI CCTV Lab 👁️

A comprehensive, containerized CCTV analysis system. It ingests RTSP streams, performs real-time AI analysis (Person Detection, Face Recognition, Gesture Recognition), and provides a live dashboard of people's locations.

## Features

-   **Multi-Camera Support**: Ingests multiple RTSP streams simultaneously.
-   **Low Latency**: Optimized ffmpeg ingest and threaded processing for <1s latency.
-   **Person Detection**: YOLOv8 (ONNX) + ByteTrack for robust tracking.
-   **Face Recognition**: InsightFace for identifying known individuals.
-   **Gesture Recognition**: MediaPipe for detecting hand gestures (e.g., waving).
-   **Unknowns Review**: "Human-in-the-loop" web interface to label unknown faces.
-   **Location Tracking**: Real-time dashboard showing the last seen location of every person.
-   **Dockerized**: Full stack (Ingest, Analysis, DB, Web, API) deployed via Docker Compose.

## Architecture

1.  **Ingest (`src/run_ingest.py`)**: Reads camera streams and re-publishes them to a local RTSP server (MediaMTX) for stable consumption.
2.  **Analysis (`src/analyse_rtsp.py`)**: Consumes streams, runs AI models, and updates the PostgreSQL database.
3.  **API (`src/api.py`)**: FastAPI service providing location data.
4.  **Identity Web App (`src/identity_web_app.py`)**: Flask app for managing identities.
5.  **Dashboard**: Static HTML/JS frontend served via Nginx.
6.  **Database**: PostgreSQL for storing identities, sightings, and location logs.

## Quick Start

### Prerequisites
-   Docker & Docker Compose
-   **For GPU Support**: NVIDIA Drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed.

### Configuration
Edit `cameras.yml` to define your RTSP sources:

```yaml
cameras:
  - name: lobby
    url: rtsp://192.168.1.100/stream1
  - name: office
    url: rtsp://192.168.1.101/stream1
```

### Run (GPU / Default)
The default configuration is optimized for NVIDIA GPUs (Grace Blackwell, RTX, etc.).
```bash
docker-compose up --build -d
```

### Run (CPU Only)
To run on systems without an NVIDIA GPU (e.g., Mac M-series, standard Linux servers):
```bash
docker-compose -f docker-compose.cpu.yml up --build -d
```

### Access

-   **Location Dashboard**: [http://localhost](http://localhost)
-   **Identity Management**: [http://localhost:5001](http://localhost:5001)

## Development (Local)

To run without Docker (requires Python 3.10+, PostgreSQL running locally):

1.  **Install Dependencies**:
    *   **GPU**: `pip install -r requirements.gpu.txt`
    *   **CPU**: `pip install -r requirements.cpu.txt`
2.  **Run Ingest**: `python src/run_ingest.py cameras.yml`
3.  **Run Analysis**: `python src/run_analysis.py cameras.yml`
4.  **Run API**: `uvicorn src.api:app --reload`
5.  **Run Web App**: `python src/identity_web_app.py`

## Migration / Backup & Restore

To move the system to another machine or back up your data:

### 1. On Source Machine

1.  **Stop Containers**:
    ```bash
    docker-compose down
    ```
2.  **Backup Database**:
    Use a temporary container to dump the database to a file.
    ```bash
    docker-compose up -d postgres
    docker exec cctv-postgres pg_dump -U user -d cctv > cctv_backup.sql
    docker-compose down
    ```
3.  **Copy Files**:
    Transfer the entire project directory (including `src`, `cameras.yml`, `cctv_backup.sql`, and the `data` folder) to the new machine.
    *   Note: The `data` folder contains face crops and identity images.

### 2. On Target Machine

1.  **Prerequisites**: Ensure Docker and Docker Compose are installed.
2.  **Restore Files**: Place the project directory on the new machine.
3.  **Start Database**:
    ```bash
    docker-compose up -d postgres
    ```
4.  **Restore Database**:
    Wait for Postgres to accept connections, then:
    ```bash
    cat cctv_backup.sql | docker exec -i cctv-postgres psql -U user -d cctv
    ```
5.  **Start System**:
    ```bash
    docker-compose up -d --build
    ```

## License
MIT
