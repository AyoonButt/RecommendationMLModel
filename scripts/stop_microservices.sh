#!/bin/bash

# Stop Microservices Script
# This script stops all running microservices

echo "Stopping Recommendation System Microservices..."

# Function to stop service by PID file
stop_service() {
    local service_name=$1
    local pid_file="${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $service_name (PID: $pid)..."
            kill $pid
            sleep 2
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "Force stopping $service_name..."
                kill -9 $pid
            fi
            
            echo "✓ $service_name stopped"
        else
            echo "✓ $service_name was not running"
        fi
        rm -f "$pid_file"
    else
        echo "No PID file found for $service_name"
    fi
}

# Stop services in reverse order
stop_service "core-recommendations"
stop_service "social-recommendations" 
stop_service "comment-analysis"

# Also kill any remaining Python processes running our services
echo "Cleaning up any remaining service processes..."

pkill -f "core_recommendations_service.py" 2>/dev/null
pkill -f "social_recommendations_service.py" 2>/dev/null
pkill -f "comment_analysis_service.py" 2>/dev/null

echo ""
echo "🛑 All microservices stopped"
echo ""
echo "To restart services, run: ./start_microservices.sh"