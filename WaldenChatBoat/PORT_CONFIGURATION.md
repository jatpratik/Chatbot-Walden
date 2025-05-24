# Port Configuration Guide

This guide explains how to change the port numbers for the frontend and API services in the Walden Chatbot application.

## Changing Port Numbers

### 1. Update docker-compose.yml

The primary place to change port numbers is in the `docker-compose.yml` file:

```yaml
services:
  walden_api:
    # ... other configurations
    ports:
      - "9000:8000"  # Format: "HOST_PORT:CONTAINER_PORT"
    # ... other configurations
    
  web:
    # ... other configurations
    ports:
      - "3000:80"  # Format: "HOST_PORT:CONTAINER_PORT"
    # ... other configurations
```

- For the API service (`walden_api`):
  - The format is `"HOST_PORT:CONTAINER_PORT"`
  - Change only the HOST_PORT (left side)
  - Keep the CONTAINER_PORT (right side) as 8000 unless you modify the API code

- For the web service:
  - The format is `"HOST_PORT:CONTAINER_PORT"`
  - Change only the HOST_PORT (left side)
  - Keep the CONTAINER_PORT (right side) as 80 (standard Nginx port)

### 2. Update README.md

After changing the ports, remember to update the README.md file to reflect the new port numbers:

```markdown
2. Access the application:
   - Frontend: http://localhost:3000
   - API: http://localhost:3000/api
   - Weaviate: http://localhost:8089
```

### 3. No Changes Needed in index.html

The frontend code in `index.html` is already using a relative URL for API calls:

```javascript
const response = await fetch('/api/chat', {
    // ... other configurations
});
```

This means it will automatically work with any port number you set for the web service.

## Example Port Configurations

### Default Configuration
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API through Nginx: http://localhost:8501/api
- Weaviate: http://localhost:8089

### Custom Configuration Example
- Frontend: http://localhost:3000
- API: http://localhost:9000
- API through Nginx: http://localhost:3000/api
- Weaviate: http://localhost:8089

## Applying Changes

After changing the port numbers, restart the Docker containers:

```bash
docker-compose down
docker-compose up -d
```

Then access your application using the new port numbers.
