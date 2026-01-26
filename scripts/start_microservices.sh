#!/bin/bash

# Start Microservices Script
# This script starts all three microservices for local development

echo "Starting Recommendation System Microservices..."

# Load environment variables
if [ -f ../.env.microservices ]; then
    echo "Loading environment variables from ../.env.microservices..."
    set -a  # Mark variables for export
    source ../.env.microservices 2>/dev/null || echo "Note: Some env vars may not be valid"
    set +a  # Stop marking variables for export
else
    echo "Warning: ../.env.microservices not found, using default values"
fi

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Function to start service in background
start_service() {
    local service_name=$1
    local script_name=$2
    local port=$3
    
    echo "Starting $service_name on port $port..."
    
    if check_port $port; then
        nohup python $script_name > ../logs/${service_name}.log 2>&1 &
        echo $! > ${service_name}.pid
        sleep 2
        
        if ps -p $! > /dev/null; then
            echo "✓ $service_name started successfully (PID: $!)"
        else
            echo "✗ Failed to start $service_name"
            return 1
        fi
    else
        echo "✗ Cannot start $service_name - port $port is in use"
        return 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo "✓ $service_name is ready"
            return 0
        else
            echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
            sleep 2
            attempt=$((attempt + 1))
        fi
    done
    
    echo "✗ $service_name failed to start within timeout"
    return 1
}

# Create logs directory
mkdir -p ../logs

# Start Redis if not running
echo "Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis not running. Please start Redis first:"
    echo "  brew services start redis  # On macOS"
    echo "  sudo systemctl start redis  # On Linux"
    echo "  Or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    exit 1
else
    echo "✓ Redis is running"
fi

# Check if Spring API is running on port 8080, if not start Comment Analysis Service
if curl -s -f "http://localhost:8080/actuator/health" > /dev/null 2>&1; then
    echo "✓ Spring API detected on port 8080 - Comment Analysis will integrate with it"
    # Set comment service to use Spring API endpoints
    export COMMENT_SERVICE_INTEGRATED=true
else
    echo "Spring API not detected - Starting standalone Comment Analysis Service on port ${COMMENT_SERVICE_PORT:-8080}"
    start_service "comment-analysis" "../services/comment-analysis/comment_analysis_service.py" ${COMMENT_SERVICE_PORT:-8080}
    wait_for_service "Comment Analysis Service" "http://localhost:${COMMENT_SERVICE_PORT:-8080}/health"
fi

# Start Social Recommendations Service (Port 8081) 
start_service "social-recommendations" "../services/social-recommendations/social_recommendations_service.py" ${SOCIAL_SERVICE_PORT:-8081}
wait_for_service "Social Recommendations Service" "http://localhost:${SOCIAL_SERVICE_PORT:-8081}/health"

# Start Core Recommendations Service (Port 5000)
start_service "core-recommendations" "../services/core-recommendations/core_recommendations_service.py" ${CORE_SERVICE_PORT:-5000}
wait_for_service "Core Recommendations Service" "http://localhost:${CORE_SERVICE_PORT:-5000}/health"

echo ""
echo "🚀 All microservices started successfully!"
echo ""
echo "Service endpoints:"
echo "  Core Recommendations:  http://localhost:${CORE_SERVICE_PORT:-5000}"
echo "  Social Recommendations: http://localhost:${SOCIAL_SERVICE_PORT:-8081}"
echo "  Comment Analysis:      http://localhost:${COMMENT_SERVICE_PORT:-8080}"
echo ""
echo "Health checks:"
echo "  curl http://localhost:${CORE_SERVICE_PORT:-5000}/health"
echo "  curl http://localhost:${SOCIAL_SERVICE_PORT:-8081}/health"
echo "  curl http://localhost:${COMMENT_SERVICE_PORT:-8080}/health"
echo ""
echo "To stop all services, run: ./scripts/stop_microservices.sh"
echo "Logs are available in the logs/ directory"