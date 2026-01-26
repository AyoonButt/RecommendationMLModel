#!/bin/bash

# Stop Services Manual Script

echo "🛑 Stopping Recommendation Services..."

# Function to stop service by PID
stop_service_simple() {
    local service_name=$1
    local pid_file="/tmp/${service_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "🔴 Stopping $service_name (PID: $pid)..."
            kill $pid
            sleep 1
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "🔴 Force stopping $service_name..."
                kill -9 $pid
            fi
            
            echo "✅ $service_name stopped"
        else
            echo "ℹ️ $service_name was not running"
        fi
        rm -f "$pid_file"
    else
        echo "ℹ️ No PID file found for $service_name"
    fi
}

# Stop services
stop_service_simple "core-recommendations"
stop_service_simple "social-recommendations"
stop_service_simple "comment-analysis"

# Clean up any remaining processes
echo "🧹 Cleaning up remaining processes..."
pkill -f "core_recommendations_service.py" 2>/dev/null
pkill -f "social_recommendations_service.py" 2>/dev/null  
pkill -f "comment_analysis_service.py" 2>/dev/null

echo ""
echo "✅ All services stopped"
echo "🚀 To restart: ./start_services_manual.sh"