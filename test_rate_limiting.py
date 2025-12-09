#!/usr/bin/env python3
"""
Test script for validating the rate limiting improvements in ultimatum_game_parallel.py

This script runs a series of tests with different configurations to verify that
the rate limiting and error handling improvements work correctly.
"""

import subprocess
import sys
import time

def run_test(test_name, command):
    """Run a single test configuration."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"Command: {command}")
    print('='*70)

    start = time.time()
    try:
        # Run the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )

        elapsed = time.time() - start

        # Check for specific error patterns
        output = result.stdout + result.stderr
        rate_limit_errors = output.count("Rate limit")
        list_index_errors = output.count("list index out of range")
        successful = "SIMULATION COMPLETE" in output

        print(f"\n✅ Test completed in {elapsed:.1f}s")
        print(f"   Rate limit errors: {rate_limit_errors}")
        print(f"   List index errors: {list_index_errors}")
        print(f"   Successful: {'Yes' if successful else 'No'}")

        if list_index_errors > 0:
            print("   ⚠️  WARNING: List index errors detected - bug may not be fully fixed")

        if not successful and elapsed < 100:
            print("   ⚠️  WARNING: Simulation may have failed")

        return successful, rate_limit_errors, list_index_errors

    except subprocess.TimeoutExpired:
        print(f"\n❌ Test timed out after 120 seconds")
        return False, 0, 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False, 0, 0


def main():
    """Run a series of tests with different configurations."""
    print("="*70)
    print("ULTIMATUM GAME PARALLEL - RATE LIMITING TEST SUITE")
    print("="*70)

    tests = [
        # Test 1: Conservative settings (should work well)
        ("Conservative: 5 workers, 5 API limit",
         "python3 ultimatum_game_parallel.py -p 0,1,2 -g 2 -w 5 --api-rate-limit 5"),

        # Test 2: Moderate settings
        ("Moderate: 10 workers, 10 API limit",
         "python3 ultimatum_game_parallel.py -p 0,1,2,3 -g 2 -w 10 --api-rate-limit 10"),

        # Test 3: Aggressive but with rate limiting
        ("Aggressive with limiting: 20 workers, 10 API limit",
         "python3 ultimatum_game_parallel.py -p 0,1,2,3 -g 3 -w 20 --api-rate-limit 10"),

        # Test 4: Very conservative (for comparison)
        ("Very Conservative: 3 workers, 3 API limit",
         "python3 ultimatum_game_parallel.py -p 0,1 -g 2 -w 3 --api-rate-limit 3"),
    ]

    results = []

    for test_name, command in tests:
        success, rate_limits, index_errors = run_test(test_name, command)
        results.append((test_name, success, rate_limits, index_errors))

        # Small delay between tests
        time.sleep(2)

    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)

    all_passed = True
    for test_name, success, rate_limits, index_errors in results:
        status = "✅ PASS" if success and index_errors == 0 else "❌ FAIL"
        if not success or index_errors > 0:
            all_passed = False

        print(f"{status} - {test_name}")
        if rate_limits > 0:
            print(f"      (Hit {rate_limits} rate limits but recovered)")
        if index_errors > 0:
            print(f"      ⚠️  Had {index_errors} list index errors")

    print(f"\n{'='*70}")
    if all_passed:
        print("✅ ALL TESTS PASSED - Rate limiting is working correctly!")
        print("\nRecommended settings for your use case:")
        print("  - For fast execution: --workers 10 --api-rate-limit 10")
        print("  - For large simulations: --workers 20 --api-rate-limit 10")
        print("  - For maximum speed (risky): --workers 30 --api-rate-limit 15")
    else:
        print("⚠️  SOME TESTS FAILED - Check the output above for details")
        print("\nThe 'list index out of range' errors indicate the bug fix may need review.")

    print('='*70)


if __name__ == "__main__":
    main()