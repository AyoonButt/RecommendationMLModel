#!/usr/bin/env python3
"""
Test RL Integration
Quick test to verify the RL integration is working correctly.
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("rl-integration-test")

def test_imports():
    """Test that RL components can be imported correctly."""
    logger.info("Testing imports...")
    
    try:
        # Test RL-agent components
        sys.path.append('services/rl-agent')
        from RLIntegration import create_rl_integration_manager
        logger.info("✅ RL-agent components imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import RL-agent components: {e}")
        return False
    
    try:
        # Test shared components
        sys.path.append('shared/components')
        from RLEnhancedMetadataEnhancer import create_rl_enhanced_metadata_enhancer
        logger.info("✅ RLEnhancedMetadataEnhancer imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import RLEnhancedMetadataEnhancer: {e}")
        return False
    
    try:
        # Test simple RL metadata
        sys.path.append('services/core-recommendations')
        from simple_rl_metadata import RLMetadataEnhancer
        logger.info("✅ Simple RL metadata imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import simple RL metadata: {e}")
        return False
    
    return True

def test_rl_metadata_creation():
    """Test creating RL metadata enhancer."""
    logger.info("Testing RL metadata enhancer creation...")
    
    try:
        sys.path.append('services/core-recommendations')
        from simple_rl_metadata import create_rl_metadata_enhancer
        
        # Create enhancer
        enhancer = create_rl_metadata_enhancer("http://localhost:8080")
        logger.info(f"✅ Created RL metadata enhancer: {type(enhancer).__name__}")
        
        # Test stats
        stats = enhancer.get_stats()
        logger.info(f"✅ Retrieved RL stats: {stats.get('rl_enhancement', {}).keys()}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create RL metadata enhancer: {e}")
        return False

def test_core_service_integration():
    """Test that core service can import RL components."""
    logger.info("Testing core service integration...")
    
    try:
        # Change to services directory to simulate service startup
        original_path = sys.path.copy()
        sys.path.insert(0, 'services/core-recommendations')
        sys.path.insert(0, 'shared/components')
        
        # Import the updated core service
        from simple_rl_metadata import RLMetadataEnhancer
        
        # Test creation like the service does
        enhancer = RLMetadataEnhancer("http://localhost:8080")
        logger.info("✅ Core service can create RL-enhanced MetadataEnhancer")
        
        # Restore path
        sys.path = original_path
        return True
        
    except Exception as e:
        logger.error(f"❌ Core service integration failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting RL Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("RL Metadata Creation", test_rl_metadata_creation),
        ("Core Service Integration", test_core_service_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 30)
        
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RL integration is working correctly.")
        print("\n🎯 Next Steps:")
        print("1. Start your core-recommendations service")
        print("2. Test with real recommendation requests")
        print("3. Send user interaction data to /interactions endpoint")
        print("4. Monitor RL stats at /stats endpoint")
    else:
        logger.error("⚠️  Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())