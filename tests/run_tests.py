#!/usr/bin/env python3
"""
Test runner script for KestrelAI with service checking and categorization.
"""

import sys
import os
import subprocess
import argparse
from typing import List, Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.utils.check_services import check_services_for_testing
from tests.utils.test_config import get_test_config, get_service_requirements, TEST_CATEGORIES


def run_pytest(args: List[str], verbose: bool = True) -> int:
    """Run pytest with given arguments."""
    cmd = ["python", "-m", "pytest"] + args
    
    if verbose:
        cmd.append("-v")
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def check_services_before_test(categories: List[str]) -> bool:
    """Check if required services are available before running tests."""
    print("🔍 Checking required services...")
    
    all_required_services = set()
    for category in categories:
        required = get_service_requirements(category)
        all_required_services.update(required)
    
    if not all_required_services:
        print("✅ No external services required for these tests.")
        return True
    
    # Check services
    all_available, available_services, unavailable_services = check_services_for_testing()
    
    missing_services = [s for s in all_required_services if s not in available_services]
    
    if missing_services:
        print(f"❌ Missing required services: {', '.join(missing_services)}")
        print("Please start the required services before running tests.")
        return False
    
    print("✅ All required services are available.")
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="KestrelAI Test Runner")
    parser.add_argument(
        "categories",
        nargs="*",
        choices=list(TEST_CATEGORIES.keys()),
        help="Test categories to run (default: all)"
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Skip service availability check"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    args = parser.parse_args()
    
    # Determine test categories
    if args.categories:
        categories = args.categories
    else:
        categories = list(TEST_CATEGORIES.keys())
    
    # Check services if not skipped
    if not args.skip_service_check:
        if not check_services_before_test(categories):
            sys.exit(1)
    
    # Build pytest arguments
    pytest_args = []
    
    # Add category markers
    for category in categories:
        pytest_args.extend(["-m", category])
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(["--cov=KestrelAI", "--cov-report=html", "--cov-report=term"])
    
    # Add parallel execution if requested
    if args.parallel:
        pytest_args.extend(["-n", "auto"])
    
    # Set verbosity
    if args.quiet:
        pytest_args.append("-q")
    elif args.verbose:
        pytest_args.append("-v")
    
    # Add test directory
    pytest_args.append("tests/")
    
    # Run tests
    print(f"🧪 Running tests for categories: {', '.join(categories)}")
    exit_code = run_pytest(pytest_args, verbose=args.verbose)
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
