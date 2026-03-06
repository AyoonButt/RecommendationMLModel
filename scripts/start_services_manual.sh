#!/bin/bash

# Manual Service Startup (No Redis Required)
# This script starts services with MockRedis for development

echo "🚀 Starting Recommendation Services (Development Mode)"
echo "================================================="

# Set up environment for development
export LOCAL_DEV=true
export REDIS_HOST=localhost
export REDIS_PORT=6379
export DEBUG=true
export SERVICE_AUTH_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJtbC1zZXJ2aWNlIiwianRpIjoibWwtc2VydmljZS10b2tlbi0wMDMiLCJpYXQiOjE3NjE3MjA0MDQsImV4cCI6MTc5MzI1NjQwNCwidXNlcklkIjotMSwiYXV0aG9yaXRpZXMiOlsiUk9MRV9TRVJWSUNFIl0sInRva2VuVHlwZSI6ImFjY2VzcyJ9.yx3avxLlUAL7RDygce_aAuIP8IS5a9Qk5QLhboz6B4qUhyxfVqMKx38KL4AmvFgzp0p-wlbq9PpnhQjFcprZWg"

# Service ports
export CORE_SERVICE_PORT=5000
export SOCIAL_SERVICE_PORT=8081
export COMMENT_SERVICE_PORT=8082
# SPRING_API_URL loaded from .env

echo "📋 Configuration:"
echo "  Core Service: Port $CORE_SERVICE_PORT"
echo "  Social Service: Port $SOCIAL_SERVICE_PORT" 
echo "  Comment Service: Port $COMMENT_SERVICE_PORT"
echo "  Spring API: $SPRING_API_URL"
echo ""

# Function to start service in background
start_service_simple() {
    local service_name=$1
    local service_dir=$2
    local port=$3
    
    echo "🔧 Starting $service_name on port $port..."
    
    # Check if port is available
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "❌ Port $port is already in use"
        return 1
    fi
    
    # Start service
    local script_file="${service_dir}/${service_name//-/_}_service.py"
    if [ -f "$script_file" ]; then
        echo "📁 Running: python3 $script_file"
        python3 "$script_file" > "/tmp/${service_name}.log" 2>&1 &
        service_pid=$!
        echo $service_pid > /tmp/${service_name}.pid
        
        echo "✅ $service_name started (PID: $service_pid)"
        sleep 2
    else
        echo "❌ Script not found: $script_file"
        return 1
    fi
    
    return 0
}

# Start Comment Analysis Service
echo "1️⃣ Starting Comment Analysis Service..."
start_service_simple "comment-analysis" "../services/comment-analysis" $COMMENT_SERVICE_PORT

# Start Social Recommendations Service  
echo "2️⃣ Starting Social Recommendations Service..."
start_service_simple "social-recommendations" "../services/social-recommendations" $SOCIAL_SERVICE_PORT

# Start Core Recommendations Service
echo "3️⃣ Starting Core Recommendations Service..."
start_service_simple "core-recommendations" "../services/core-recommendations" $CORE_SERVICE_PORT

echo ""
echo "🎉 Services Starting Complete!"
echo ""
echo "🌐 Service URLs:"
echo "  Core Recommendations:   http://localhost:$CORE_SERVICE_PORT"
echo "  Social Recommendations: http://localhost:$SOCIAL_SERVICE_PORT"
echo "  Comment Analysis:       http://localhost:$COMMENT_SERVICE_PORT"
echo ""
echo "🧪 Test Commands:"
echo "  curl http://localhost:$CORE_SERVICE_PORT/health"
echo "  curl http://localhost:$SOCIAL_SERVICE_PORT/health"
echo "  curl http://localhost:$COMMENT_SERVICE_PORT/health"
echo ""
echo "🛑 To stop services: ./stop_services_manual.sh"
echo "📊 View logs: tail -f /tmp/*.log"

# Wait a moment for services to initialize
sleep 3

# Test health endpoints
echo "🔍 Testing service health..."
for port in $CORE_SERVICE_PORT $SOCIAL_SERVICE_PORT $COMMENT_SERVICE_PORT; do
    if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "✅ Service on port $port is healthy"
    else
        echo "⚠️ Service on port $port may still be starting..."
    fi
done

echo ""
echo "🎯 Ready for testing! Services are running in background."