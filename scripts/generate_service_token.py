#!/usr/bin/env python3
"""
Generate Service JWT Token for ML Services
Creates a proper JWT token that matches your Spring Security expectations
"""

import sys
import os
import json
import time
import base64
import hmac
import hashlib
import requests
from typing import Optional

# Add the shared directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

try:
    from auth.ServiceTokenManager import ServiceTokenManager, setup_service_authentication
except ImportError:
    print("Could not import ServiceTokenManager. Make sure the shared/auth directory exists.")
    sys.exit(1)

def create_jwt_token(secret_key: str, service_name: str = "ml-service", 
                    expires_in_hours: int = 8760) -> str:
    """
    Create a JWT token manually
    
    Args:
        secret_key: JWT secret key from your Spring application
        service_name: Name of the service
        expires_in_hours: Expiry time in hours (default: 1 year)
        
    Returns:
        Complete JWT token string
    """
    # JWT Header
    header = {
        "typ": "JWT",
        "alg": "HS512"
    }
    
    # JWT Payload
    current_time = int(time.time())
    payload = {
        "sub": service_name,
        "userId": -1,  # Special service user ID
        "authorities": ["ROLE_SERVICE"],
        "tokenType": "access",
        "iat": current_time,
        "exp": current_time + (expires_in_hours * 3600)
    }
    
    # Encode header and payload
    header_encoded = base64.urlsafe_b64encode(
        json.dumps(header, separators=(',', ':')).encode()
    ).decode().rstrip('=')
    
    payload_encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(',', ':')).encode()
    ).decode().rstrip('=')
    
    # Create signature
    message = f"{header_encoded}.{payload_encoded}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha512
    ).digest()
    
    signature_encoded = base64.urlsafe_b64encode(signature).decode().rstrip('=')
    
    # Complete JWT token
    jwt_token = f"{header_encoded}.{payload_encoded}.{signature_encoded}"
    
    return jwt_token

def get_jwt_secret_from_spring() -> Optional[str]:
    """
    Try to get JWT secret from Spring Boot application
    """
    # Common JWT secret locations/defaults
    possible_secrets = [
        os.environ.get('JWT_SECRET'),
        'your-secret-key-here',  # Default from our generator
        '811d0bc5-48d2-4d69-9e81-959644b72f93',  # Your generated security password
        'mySecretKey',
        'springboot-jwt-secret'
    ]
    
    for secret in possible_secrets:
        if secret:
            return secret
    
    return None

def request_token_from_api(api_base_url: str) -> Optional[str]:
    """
    Try to request a token from your Spring API
    """
    try:
        # Try the dev endpoint we created
        url = f"{api_base_url}/dev/generate-service-token"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('token')
        
        print(f"API endpoint returned status: {response.status_code}")
        return None
        
    except Exception as e:
        print(f"Could not request token from API: {e}")
        return None

def main():
    """Main function to generate and configure service token"""
    api_base_url = "http://localhost:8080"
    service_name = "ml-service"
    
    print("🔐 ML Service Token Generator")
    print("=" * 40)
    
    # Initialize token manager
    token_manager = ServiceTokenManager(service_name)
    
    # Method 1: Try to get token from your Spring API
    print("🌐 Trying to request token from Spring API...")
    api_token = request_token_from_api(api_base_url)
    
    if api_token:
        print("✅ Successfully obtained token from API")
        token_manager.save_token(api_token)
    else:
        print("❌ Could not get token from API, trying manual generation...")
        
        # Method 2: Generate token manually
        jwt_secret = get_jwt_secret_from_spring()
        
        if jwt_secret:
            print(f"🔑 Using JWT secret: {jwt_secret[:10]}...")
            manual_token = create_jwt_token(jwt_secret, service_name)
            token_manager.save_token(manual_token)
            print("✅ Generated token manually")
        else:
            print("❌ Could not determine JWT secret")
            print("💡 Please provide JWT secret manually:")
            jwt_secret = input("Enter JWT secret key: ").strip()
            
            if jwt_secret:
                manual_token = create_jwt_token(jwt_secret, service_name)
                token_manager.save_token(manual_token)
                print("✅ Generated token with provided secret")
            else:
                print("❌ No secret provided, cannot generate token")
                return False
    
    # Verify token
    print("\\n📋 Token Information:")
    token_info = token_manager.get_token_info()
    for key, value in token_info.items():
        print(f"  {key}: {value}")
    
    # Test token with API
    print("\\n🧪 Testing token with API...")
    headers = token_manager.get_service_headers()
    
    if headers:
        try:
            test_url = f"{api_base_url}/api/internal/ml/users/7/candidates?limit=1&contentType=POSTS"
            response = requests.get(test_url, headers=headers, timeout=10)
            
            print(f"  Status Code: {response.status_code}")
            if response.status_code == 200:
                print("✅ Token works! API call successful")
            elif response.status_code == 401:
                print("❌ Token rejected - authentication failed")
            elif response.status_code == 403:
                print("❌ Token accepted but access forbidden - check roles")
            else:
                print(f"⚠️ Unexpected response: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
        
        except Exception as e:
            print(f"❌ Error testing token: {e}")
    
    # Generate environment variable export
    auth_header = token_manager.get_authorization_header()
    if auth_header:
        token = auth_header.split(' ', 1)[1]  # Remove "Bearer " prefix
        print("\\n🔧 Environment Setup:")
        print(f"export SERVICE_AUTH_TOKEN='{token}'")
        print("\\n💡 Add this to your ML service startup script!")
        
        # Update the startup script
        try:
            script_path = os.path.join(os.path.dirname(__file__), 'start_services_manual.sh')
            if os.path.exists(script_path):
                # Read current script
                with open(script_path, 'r') as f:
                    content = f.read()
                
                # Replace the token
                if 'SERVICE_AUTH_TOKEN=' in content:
                    import re
                    content = re.sub(
                        r'export SERVICE_AUTH_TOKEN="[^"]*"',
                        f'export SERVICE_AUTH_TOKEN="{token}"',
                        content
                    )
                    
                    with open(script_path, 'w') as f:
                        f.write(content)
                    
                    print(f"✅ Updated startup script: {script_path}")
                else:
                    print(f"⚠️ Could not find SERVICE_AUTH_TOKEN in {script_path}")
        
        except Exception as e:
            print(f"⚠️ Could not update startup script: {e}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\\n🎉 Service token setup complete!")
        else:
            print("\\n❌ Service token setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n👋 Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")
        sys.exit(1)