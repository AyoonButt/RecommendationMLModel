#!/usr/bin/env python3
"""
Service Token Manager for ML Microservices
Manages JWT tokens for service-to-service authentication
Similar to Android TokenManager but for Python services
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("service-token-manager")

@dataclass
class ServiceTokenInfo:
    access_token: str
    token_type: str = "Bearer"
    expires_at: Optional[int] = None
    service_name: str = "ml-service"
    roles: list = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = ["ROLE_SERVICE"]

class ServiceTokenManager:
    """
    Manages JWT tokens for ML service authentication
    Handles token storage, validation, and refresh
    """
    
    def __init__(self, service_name: str = "ml-service", cache_file: str = None):
        self.service_name = service_name
        self.token_buffer_time = 5 * 60  # 5 minutes buffer before expiry
        
        # Set up cache file path
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            cache_dir = Path.home() / ".ml-service" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / f"{service_name}_tokens.json"
        
        self._token_info: Optional[ServiceTokenInfo] = None
        self._load_cached_token()
        
        logger.info(f"ServiceTokenManager initialized for {service_name}")
    
    def save_token(self, access_token: str, token_type: str = "Bearer", 
                   expires_in_seconds: Optional[int] = None, roles: list = None) -> None:
        """
        Save service token information
        
        Args:
            access_token: JWT access token
            token_type: Token type (usually "Bearer")
            expires_in_seconds: Token expiry in seconds from now
            roles: List of roles for the service
        """
        expires_at = None
        if expires_in_seconds:
            expires_at = int(time.time()) + expires_in_seconds
        else:
            # Default to 1 year if no expiry provided
            expires_at = int(time.time()) + (365 * 24 * 60 * 60)
        
        if roles is None:
            roles = ["ROLE_SERVICE"]
        
        self._token_info = ServiceTokenInfo(
            access_token=access_token,
            token_type=token_type,
            expires_at=expires_at,
            service_name=self.service_name,
            roles=roles
        )
        
        self._save_to_cache()
        logger.info(f"Saved token for service {self.service_name} (expires: {expires_at})")
    
    def get_access_token(self) -> Optional[str]:
        """Get the current access token"""
        if self._token_info and not self.is_token_expired():
            return self._token_info.access_token
        return None
    
    def get_authorization_header(self) -> Optional[str]:
        """Get the full Authorization header value"""
        token = self.get_access_token()
        if token and self._token_info:
            return f"{self._token_info.token_type} {token}"
        return None
    
    def get_service_headers(self) -> Dict[str, str]:
        """Get all headers needed for service authentication"""
        headers = {}
        
        auth_header = self.get_authorization_header()
        if auth_header:
            headers['Authorization'] = auth_header
            headers['X-Service-Role'] = 'SERVICE'
            headers['X-Service-Name'] = self.service_name
        
        return headers
    
    def has_valid_token(self) -> bool:
        """Check if we have a valid, non-expired token"""
        return (self._token_info is not None and 
                self._token_info.access_token is not None and 
                not self.is_token_expiring_soon())
    
    def is_token_expired(self) -> bool:
        """Check if the current token is expired"""
        if not self._token_info or not self._token_info.expires_at:
            return True
        return time.time() >= self._token_info.expires_at
    
    def is_token_expiring_soon(self) -> bool:
        """Check if the token is expiring within the buffer time"""
        if not self._token_info or not self._token_info.expires_at:
            return True
        return time.time() >= (self._token_info.expires_at - self.token_buffer_time)
    
    def get_token_expiry_time(self) -> Optional[int]:
        """Get token expiry timestamp"""
        if self._token_info:
            return self._token_info.expires_at
        return None
    
    def clear_token(self) -> None:
        """Clear all token information"""
        self._token_info = None
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info(f"Cleared token for service {self.service_name}")
    
    def request_service_token(self, api_base_url: str, username: str = None, password: str = None) -> bool:
        """
        Request a new service token from the auth API
        
        Args:
            api_base_url: Base URL of the Spring API
            username: Service username (optional)
            password: Service password (optional)
            
        Returns:
            True if token was successfully obtained
        """
        try:
            # Option 1: Try to get token from service endpoint
            if self._try_service_token_endpoint(api_base_url):
                return True
            
            # Option 2: Try basic auth if username/password provided
            if username and password:
                return self._try_basic_auth_token(api_base_url, username, password)
            
            # Option 3: Try using environment variables
            env_username = os.environ.get('SERVICE_USERNAME')
            env_password = os.environ.get('SERVICE_PASSWORD')
            if env_username and env_password:
                return self._try_basic_auth_token(api_base_url, env_username, env_password)
            
            logger.warning("No valid authentication method available for token request")
            return False
            
        except Exception as e:
            logger.error(f"Error requesting service token: {e}")
            return False
    
    def _try_service_token_endpoint(self, api_base_url: str) -> bool:
        """Try to get token from a dedicated service token endpoint"""
        try:
            url = f"{api_base_url}/dev/generate-service-token"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                token = data.get('token')
                if token:
                    self.save_token(token)
                    logger.info("Successfully obtained service token from dedicated endpoint")
                    return True
            
        except Exception as e:
            logger.debug(f"Service token endpoint not available: {e}")
        
        return False
    
    def _try_basic_auth_token(self, api_base_url: str, username: str, password: str) -> bool:
        """Try to get token using basic authentication"""
        try:
            # This would depend on your specific auth endpoint
            url = f"{api_base_url}/auth/login"
            payload = {
                "username": username,
                "password": password
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                access_token = data.get('accessToken')
                expires_in = data.get('expiresIn')
                
                if access_token:
                    self.save_token(access_token, expires_in_seconds=expires_in)
                    logger.info("Successfully obtained service token via basic auth")
                    return True
            
        except Exception as e:
            logger.debug(f"Basic auth token request failed: {e}")
        
        return False
    
    def _load_cached_token(self) -> None:
        """Load token from cache file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                
                self._token_info = ServiceTokenInfo(
                    access_token=data['access_token'],
                    token_type=data.get('token_type', 'Bearer'),
                    expires_at=data.get('expires_at'),
                    service_name=data.get('service_name', self.service_name),
                    roles=data.get('roles', ['ROLE_SERVICE'])
                )
                
                logger.debug(f"Loaded cached token for {self.service_name}")
        
        except Exception as e:
            logger.warning(f"Could not load cached token: {e}")
            self._token_info = None
    
    def _save_to_cache(self) -> None:
        """Save token to cache file"""
        try:
            if self._token_info:
                data = {
                    'access_token': self._token_info.access_token,
                    'token_type': self._token_info.token_type,
                    'expires_at': self._token_info.expires_at,
                    'service_name': self._token_info.service_name,
                    'roles': self._token_info.roles,
                    'cached_at': int(time.time())
                }
                
                # Ensure directory exists
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.cache_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.debug(f"Saved token to cache: {self.cache_file}")
        
        except Exception as e:
            logger.warning(f"Could not save token to cache: {e}")
    
    def get_token_info(self) -> Dict[str, Any]:
        """Get comprehensive token information for debugging"""
        if not self._token_info:
            return {"status": "no_token"}
        
        return {
            "status": "valid" if self.has_valid_token() else "invalid",
            "service_name": self._token_info.service_name,
            "token_type": self._token_info.token_type,
            "expires_at": self._token_info.expires_at,
            "expires_in_seconds": (self._token_info.expires_at - int(time.time())) if self._token_info.expires_at else None,
            "is_expired": self.is_token_expired(),
            "is_expiring_soon": self.is_token_expiring_soon(),
            "roles": self._token_info.roles,
            "has_token": bool(self._token_info.access_token)
        }


# Global service token manager instance
_service_token_manager = None

def get_service_token_manager(service_name: str = "ml-service") -> ServiceTokenManager:
    """Get the global service token manager instance"""
    global _service_token_manager
    if _service_token_manager is None:
        _service_token_manager = ServiceTokenManager(service_name)
    return _service_token_manager


def setup_service_authentication(api_base_url: str, service_name: str = "ml-service") -> bool:
    """
    Set up service authentication by requesting a token
    
    Args:
        api_base_url: Base URL of the Spring API
        service_name: Name of the service
        
    Returns:
        True if authentication was successfully set up
    """
    token_manager = get_service_token_manager(service_name)
    
    # Check if we already have a valid token
    if token_manager.has_valid_token():
        logger.info(f"Service {service_name} already has valid authentication")
        return True
    
    # Try to get a new token
    if token_manager.request_service_token(api_base_url):
        logger.info(f"Successfully set up authentication for service {service_name}")
        return True
    
    logger.error(f"Failed to set up authentication for service {service_name}")
    return False


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Create token manager
    manager = ServiceTokenManager("ml-service")
    
    # Example: manually set a token for testing
    test_token = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.test.token"
    manager.save_token(test_token, expires_in_seconds=3600)
    
    # Test token operations
    print("Token info:", manager.get_token_info())
    print("Auth header:", manager.get_authorization_header())
    print("Service headers:", manager.get_service_headers())
    print("Has valid token:", manager.has_valid_token())