# Docker Deployment

## Quick Start

```bash
# Clone the repo
git clone https://github.com/ENDEVSOLS/LongParser.git
cd LongParser

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and MongoDB URI

# Start all services
docker compose up
```

The API will be available at `http://localhost:8000`.

## Services

The `docker-compose.yml` includes:

| Service | Port | Description |
|---|---|---|
| `longparser` | 8000 | FastAPI server |
| `mongo` | 27017 | MongoDB for job/session storage |
| `redis` | 6379 | Task queue |

## Production Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[server]"
CMD ["uvicorn", "longparser.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Scaling

For horizontal scaling, run multiple `longparser` containers with a shared MongoDB and Redis:

```bash
docker compose up --scale longparser=3
```

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "longparser-api"}
```
