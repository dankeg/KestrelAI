#!/usr/bin/env python3
"""
Service availability checker for tests that require external services.
This utility helps determine which external services are available for testing.
"""

import requests
import sys
import os
from typing import Dict, List, Tuple


def get_service_config() -> Dict[str, Dict[str, str]]:
    """Get service configuration with URLs and expected status codes."""
    return {
        "Backend API": {
            "url": "http://localhost:8000/api/v1/tasks",
            "expected_status": [200, 201]
        },
        "Frontend": {
            "url": "http://localhost:5173",
            "expected_status": [200]
        },
        "Ollama": {
            "url": "http://localhost:11434/api/tags",
            "expected_status": [200]
        },
        "SearXNG": {
            "url": "http://localhost:8080/search?q=test&format=json",
            "expected_status": [200]
        },
        "Redis": {
            "url": "redis://localhost:6379",
            "expected_status": [200]  # This will be handled specially
        }
    }


def check_single_service(service_name: str, config: Dict[str, str], timeout: int = 5) -> Tuple[bool, str]:
    """Check if a single service is available."""
    url = config["url"]
    expected_status = config["expected_status"]
    
    try:
        if url.startswith("redis://"):
            # Special handling for Redis
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=timeout)
            r.ping()
            return True, "Available"
        else:
            response = requests.get(url, timeout=timeout)
            if response.status_code in expected_status:
                return True, f"Available (status {response.status_code})"
            else:
                return False, f"Not responding (status {response.status_code})"
    except Exception as e:
        return False, f"Not available ({str(e)})"


def check_services(verbose: bool = True) -> Tuple[bool, List[str], List[str]]:
    """
    Check if required services are running.
    
    Args:
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (all_available, available_services, unavailable_services)
    """
    services = get_service_config()
    
    if verbose:
        print("Checking service availability...")
    
    available_services = []
    unavailable_services = []
    
    for service_name, config in services.items():
        is_available, message = check_single_service(service_name, config)
        
        if is_available:
            available_services.append(service_name)
            if verbose:
                print(f"✅ {service_name}: {message}")
        else:
            unavailable_services.append(service_name)
            if verbose:
                print(f"❌ {service_name}: {message}")
    
    all_available = len(unavailable_services) == 0
    
    if verbose:
        if all_available:
            print(f"\n✅ All {len(available_services)} services are available!")
        else:
            print(f"\n⚠️  Warning: {len(unavailable_services)} service(s) are not available.")
            print("Tests marked with @pytest.mark.requires_services may fail.")
    
    return all_available, available_services, unavailable_services


def check_services_for_testing() -> bool:
    """Check services specifically for testing purposes."""
    all_available, available_services, unavailable_services = check_services(verbose=False)
    
    # For testing, we need at least Backend API and Frontend
    required_services = ["Backend API", "Frontend"]
    required_available = all(service in available_services for service in required_services)
    
    if not required_available:
        missing = [service for service in required_services if service not in available_services]
        print(f"❌ Required services not available: {', '.join(missing)}")
        return False
    
    return True


if __name__ == "__main__":
    all_available, available_services, unavailable_services = check_services()
    if all_available:
        sys.exit(0)
    else:
        sys.exit(1)
