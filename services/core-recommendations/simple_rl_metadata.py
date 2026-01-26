#!/usr/bin/env python3
"""
Simple RL MetadataEnhancer Integration (Development Mode)
Clean, direct replacement for MetadataEnhancer with RL capabilities.
"""

import sys
import os

# Add paths for RL components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared/components'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../rl-agent'))

# Direct imports - no fallback complexity
from RLEnhancedMetadataEnhancer import RLEnhancedMetadataEnhancer, create_rl_enhanced_metadata_enhancer


class SimpleRLMetadata:
    """
    Simplified RL MetadataEnhancer for development.
    Direct replacement with RL capabilities, no backward compatibility layers.
    """
    
    def __init__(self, api_base_url: str, redis_client=None, cache_ttl: int = 3600):
        """Initialize with RL enhancement."""
        self.enhancer = create_rl_enhanced_metadata_enhancer(
            api_base_url=api_base_url,
            redis_client=redis_client,
            cache_ttl=cache_ttl
        )
    
    def enhance_scores(self, user_id: str, post_ids: list, base_scores, 
                      candidates: list = None, content_type: str = "posts"):
        """Enhance scores with RL."""
        return self.enhancer.enhance_scores(user_id, post_ids, base_scores, candidates, content_type)
    
    def process_user_interaction(self, user_id: str, post_id: int, interaction_type: str,
                               additional_context: dict = None):
        """Process user interaction for RL learning."""
        self.enhancer.process_user_interaction(user_id, post_id, interaction_type, additional_context)
    
    def get_stats(self):
        """Get RL statistics."""
        return self.enhancer.get_rl_stats()


# For direct import replacement
RLMetadataEnhancer = SimpleRLMetadata

# Factory function
def create_rl_metadata_enhancer(api_base_url: str, redis_client=None, cache_ttl: int = 3600):
    """Create RL metadata enhancer."""
    return SimpleRLMetadata(api_base_url, redis_client, cache_ttl)


if __name__ == "__main__":
    # Test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    enhancer = create_rl_metadata_enhancer("http://localhost:8080")
    print(f"Created: {type(enhancer).__name__}")
    print("RL MetadataEnhancer ready!")