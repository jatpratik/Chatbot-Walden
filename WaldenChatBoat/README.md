# Walden University Chatbot - Docker Setup

This repository contains a Docker setup for the Walden University Chatbot, which includes the frontend, API, and Weaviate database components.

## Prerequisites

- Docker and Docker Compose installed on your system

## Configuration

The application requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key

These can be set in the `.env` file or passed directly to Docker Compose.

## Running the Application

1. Build and start the containers:

```bash
docker-compose up -d
```

2. Access the application:
   - Frontend: http://localhost:8501
   - API: http://localhost:8501/api
   - Weaviate: http://localhost:8089

## Architecture

- **Frontend**: A simple HTML/CSS/JS application served by Nginx on port 8501
- **API**: A FastAPI application that handles chat requests and interacts with the Weaviate database
- **Database**: Weaviate running in a Docker container with data persistence

## Notes

- The Nginx configuration proxies API requests from `/api/*` to the API service
- The frontend communicates with the API through the `/api` endpoint
- The API connects to the Weaviate database using the internal Docker network
- Weaviate data is persisted using a Docker volume
