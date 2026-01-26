#!/usr/bin/env python3
"""
JWT Token Utility for ML Microservices
Handles extraction and validation of JWT tokens from requests
"""

import logging
from typing import Optional, Dict
from flask import request

logger = logging.getLogger("jwt-token-util")


def extract_jwt_token() -> Optional[str]:
    """
    Extract JWT token from Flask request headers.
    
    Returns:
        JWT token string if found, None otherwise
    """
    try:
        # Check Authorization header first
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Check for X-Auth-Token header as fallback
        token = request.headers.get('X-Auth-Token', '')
        if token:
            return token
            
        logger.debug("No JWT token found in request headers")
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting JWT token: {e}")
        return None


def create_auth_headers(token: str) -> Dict[str, str]:
    """
    Create authentication headers for service-to-service calls.
    
    Args:
        token: JWT token
        
    Returns:
        Dictionary of headers
    """
    if not token:
        return {}
    
    return {
        'Authorization': f'Bearer {token}',
        'X-Service-Role': 'SERVICE',
        'Content-Type': 'application/json'
    }


def validate_token_presence() -> bool:
    """
    Check if a valid JWT token is present in the request.
    
    Returns:
        True if token is present, False otherwise
    """
    token = extract_jwt_token()
    return token is not None and len(token.strip()) > 0


def get_token_or_fallback() -> Optional[str]:
    """
    Get JWT token from request headers or fallback to environment variable.
    This provides backward compatibility during transition.
    
    Returns:
        JWT token string or None
    """
    # First try to get token from request headers
    token = extract_jwt_token()
    if token:
        return token
    
    # Fallback to environment variable for backward compatibility
    import os
    fallback_token = os.environ.get('SERVICE_AUTH_TOKEN', '')
    if fallback_token:
        logger.info("Using fallback SERVICE_AUTH_TOKEN (consider migrating to JWT)")
        return fallback_token
    
    logger.warning("No JWT token found in request or environment")
    return None


class JwtTokenRequired:
    """
    Decorator class to ensure JWT token is present in request.
    """
    
    def __init__(self, allow_fallback: bool = True):
        """
        Initialize decorator.
        
        Args:
            allow_fallback: Whether to allow fallback to SERVICE_AUTH_TOKEN
        """
        self.allow_fallback = allow_fallback
    
    def __call__(self, func):
        """Apply the decorator to a function."""
        def wrapper(*args, **kwargs):
            if self.allow_fallback:
                token = get_token_or_fallback()
            else:
                token = extract_jwt_token()
            
            if not token:
                from flask import jsonify
                return jsonify({
                    "error": "Authentication required",
                    "message": "No valid JWT token provided"
                }), 401
            
            # Add token to kwargs for the function to use
            kwargs['jwt_token'] = token
            return func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


def log_token_info():
    """Log information about the token for debugging purposes."""
    token = extract_jwt_token()
    if token:
        # Don't log the actual token for security, just its presence and length
        logger.debug(f"JWT token present, length: {len(token)}")
    else:
        logger.debug("No JWT token in request")