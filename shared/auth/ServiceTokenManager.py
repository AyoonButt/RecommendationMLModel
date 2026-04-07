#!/usr/bin/env python3
"""
Service Token Manager for ML Microservices
Simplified to use SERVICE_AUTH_TOKEN environment variable for authentication.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("service-token-manager")


class ServiceTokenManager:
    """
    Simplified token manager that reads SERVICE_AUTH_TOKEN from environment.
    Provides headers for service-to-service authentication.
    """

    def __init__(self, service_name: str = "ml-service", cache_file: str = None):
        self.service_name = service_name
        logger.info(f"ServiceTokenManager initialized for {service_name}")

    def get_access_token(self) -> Optional[str]:
        """Get the service token from environment variable."""
        return os.environ.get('SERVICE_AUTH_TOKEN')

    def get_authorization_header(self) -> Optional[str]:
        """Get the full Authorization header value."""
        token = self.get_access_token()
        if token:
            return f"Bearer {token}"
        return None

    def get_service_headers(self) -> Dict[str, str]:
        """Get all headers needed for service authentication."""
        headers = {}
        token = self.get_access_token()
        if token:
            headers['Authorization'] = f'Bearer {token}'
            headers['X-Service-Role'] = 'SERVICE'
            headers['X-Service-Name'] = self.service_name
        return headers

    def has_valid_token(self) -> bool:
        """Check if we have a valid token from environment."""
        token = self.get_access_token()
        return token is not None and len(token) > 0

    def get_token_info(self) -> Dict[str, Any]:
        """Get token information for debugging."""
        token = self.get_access_token()
        return {
            "status": "valid" if token else "no_token",
            "service_name": self.service_name,
            "token_type": "Bearer",
            "has_token": bool(token),
            "source": "SERVICE_AUTH_TOKEN environment variable"
        }


# Global service token manager instance
_service_token_manager = None

def get_service_token_manager(service_name: str = "ml-service") -> ServiceTokenManager:
    """Get the global service token manager instance."""
    global _service_token_manager
    if _service_token_manager is None:
        _service_token_manager = ServiceTokenManager(service_name)
    return _service_token_manager


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create token manager
    manager = ServiceTokenManager("ml-service")

    # Test token operations
    print("Token info:", manager.get_token_info())
    print("Auth header:", manager.get_authorization_header())
    print("Service headers:", manager.get_service_headers())
    print("Has valid token:", manager.has_valid_token())