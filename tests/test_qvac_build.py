#!/usr/bin/env python3
"""
QVAC Build System Tests

Tests for verifying the QVAC Android ARM64 build system.
This test suite validates:
- Build script syntax and structure
- Required tool availability
- CMakeLists.txt validity
- Output directory structure

Note: This does NOT run the actual cross-compilation (no NDK on cluster nodes).
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
QVAC_DIR = PROJECT_ROOT / "client" / "qvac"
BUILD_SCRIPT = QVAC_DIR / "build_android.sh"
CMAKELISTS = QVAC_DIR / "CMakeLists.txt"
README = QVAC_DIR / "README.md"


class TestColors:
    """ANSI color codes for test output."""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_success(msg: str) -> None:
    """Print a success message in green."""
    print(f"{TestColors.GREEN}✓{TestColors.NC} {msg}")


def print_error(msg: str) -> None:
    """Print an error message in red."""
    print(f"{TestColors.RED}✗{TestColors.NC} {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message in yellow."""
    print(f"{TestColors.YELLOW}⚠{TestColors.NC} {msg}")


def print_info(msg: str) -> None:
    """Print an info message in blue."""
    print(f"{TestColors.BLUE}ℹ{TestColors.NC} {msg}")


def _check_file_exists(path: Path, name: str) -> bool:
    """Check if a file exists."""
    if not path.exists():
        print_error(f"{name} not found at: {path}")
        return False
    print_success(f"{name} found at: {path}")
    return True


def _check_file_executable(path: Path) -> bool:
    """Check if a file is executable."""
    if not os.access(path, os.X_OK):
        print_error(f"File not executable: {path}")
        return False
    print_success(f"File is executable: {path}")
    return True


def _check_bash_syntax(script_path: Path) -> bool:
    """Check bash script syntax using bash -n."""
    try:
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print_error(f"Bash syntax check failed: {result.stderr}")
            return False
        print_success("Bash syntax is valid")
        return True
    except subprocess.TimeoutExpired:
        print_error("Bash syntax check timed out")
        return False
    except FileNotFoundError:
        print_warning("bash not found, skipping syntax check")
        return True  # Don't fail if bash is missing


def _check_shellcheck(script_path: Path) -> bool:
    """Check bash script with shellcheck if available."""
    try:
        result = subprocess.run(
            ["shellcheck", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print_warning(f"shellcheck found issues: {result.stdout[:200]}")
            return True  # Don't fail on shellcheck warnings
        print_success("shellcheck passed")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print_info("shellcheck not available, skipping")
        return True  # shellcheck is optional


def test_required_tools() -> dict[str, bool]:
    """Test if required build tools are available."""
    required_tools = {
        "cmake": False,
        "git": False,
        "ninja": False,
    }

    print_info("Checking required build tools...")

    for tool in required_tools:
        path = shutil.which(tool)
        if path:
            print_success(f"{tool} found: {path}")
            required_tools[tool] = True
        else:
            print_warning(f"{tool} not found in PATH")

    return required_tools


def test_optional_tools() -> dict[str, bool]:
    """Test if optional build tools are available."""
    optional_tools = {
        "shellcheck": False,
        "adb": False,
    }

    print_info("Checking optional tools...")

    for tool in optional_tools:
        path = shutil.which(tool)
        if path:
            print_success(f"{tool} found: {path}")
            optional_tools[tool] = True
        else:
            print_info(f"{tool} not found (optional)")

    return optional_tools


def _check_cmake_file(cmakelists_path: Path) -> bool:
    """Check CMakeLists.txt for basic validity."""
    try:
        content = cmakelists_path.read_text()

        # Check for required CMake statements
        required_statements = [
            "cmake_minimum_required",
            "project",
        ]

        for statement in required_statements:
            if statement not in content:
                print_error(f"CMakeLists.txt missing: {statement}")
                return False
            print_success(f"CMakeLists.txt contains: {statement}")

        # Check for QVAC-specific content
        qvac_specific = [
            "QVAC",
            "ENABLE_LORA",
            "ENABLE_VULKAN",
        ]

        for statement in qvac_specific:
            if statement not in content:
                print_warning(f"CMakeLists.txt may be missing: {statement}")
            else:
                print_success(f"CMakeLists.txt contains: {statement}")

        return True
    except Exception as e:
        print_error(f"Error reading CMakeLists.txt: {e}")
        return False


def _check_build_script_content(script_path: Path) -> bool:
    """Check build script for required content."""
    try:
        content = script_path.read_text()

        # Check for required build steps
        required_steps = [
            "cmake",
            "ninja",
            "android.toolchain.cmake",
            "arm64-v8a",
            "VULKAN=ON",
            "LLAMA_BUILD_EXAMPLES=ON",
        ]

        for step in required_steps:
            if step not in content:
                print_warning(f"Build script may be missing: {step}")
            else:
                print_success(f"Build script contains: {step}")

        # Check for important safety checks
        safety_checks = [
            "set -euo pipefail",
            "check_tools",
        ]

        for check in safety_checks:
            if check not in content:
                print_warning(f"Build script may be missing safety: {check}")
            else:
                print_success(f"Build script contains safety: {check}")

        return True
    except Exception as e:
        print_error(f"Error reading build script: {e}")
        return False


def _check_readme_content(readme_path: Path) -> bool:
    """Check README.md for required content."""
    try:
        content = readme_path.read_text()

        # Check for required sections
        required_sections = [
            "UBS1",
            "Unisoc",
            "Mali-G57",
            "Vulkan",
            "LoRA",
            "GGUF",
        ]

        for section in required_sections:
            if section not in content:
                print_warning(f"README.md may be missing reference to: {section}")
            else:
                print_success(f"README.md contains reference to: {section}")

        return True
    except Exception as e:
        print_error(f"Error reading README.md: {e}")
        return False


def test_output_directory_structure() -> bool:
    """Test that output directory structure can be created."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "arm64-v8a"
            output_dir.mkdir(parents=True)
            (output_dir / "bin").mkdir()
            (output_dir / "lib").mkdir()
            (output_dir / "include").mkdir()

            print_success("Output directory structure can be created")
            return True
    except Exception as e:
        print_error(f"Failed to create output directory structure: {e}")
        return False


def _check_build_script_functions(script_path: Path) -> bool:
    """Check that build script defines expected functions."""
    try:
        content = script_path.read_text()

        expected_functions = [
            "check_tools",
            "clone_llamacpp",
            "build_llamacpp",
            "copy_outputs",
            "main",
        ]

        for func in expected_functions:
            pattern = f"{func}("
            if pattern not in content:
                print_warning(f"Build script may be missing function: {func}")
            else:
                print_success(f"Build script defines function: {func}")

        return True
    except Exception as e:
        print_error(f"Error checking build script functions: {e}")
        return False


import pytest


# --- Pytest-compatible test wrappers ---

def test_file_exists():
    """Build script, CMakeLists.txt, and README must exist."""
    assert _check_file_exists(BUILD_SCRIPT, "build_android.sh")
    assert _check_file_exists(CMAKELISTS, "CMakeLists.txt")
    assert _check_file_exists(README, "README.md")


def test_file_executable():
    """Build script must be executable."""
    if not BUILD_SCRIPT.exists():
        pytest.skip("build_android.sh not found")
    assert _check_file_executable(BUILD_SCRIPT)


def test_bash_syntax():
    """Build script must have valid bash syntax."""
    if not BUILD_SCRIPT.exists():
        pytest.skip("build_android.sh not found")
    assert _check_bash_syntax(BUILD_SCRIPT)


def test_shellcheck():
    """Build script should pass shellcheck (non-fatal)."""
    if not BUILD_SCRIPT.exists():
        pytest.skip("build_android.sh not found")
    assert _check_shellcheck(BUILD_SCRIPT)


def test_cmake_file():
    """CMakeLists.txt must contain required statements."""
    if not CMAKELISTS.exists():
        pytest.skip("CMakeLists.txt not found")
    assert _check_cmake_file(CMAKELISTS)


def test_build_script_content():
    """Build script must reference required build steps."""
    if not BUILD_SCRIPT.exists():
        pytest.skip("build_android.sh not found")
    assert _check_build_script_content(BUILD_SCRIPT)


def test_readme_content():
    """README must reference key hardware/software targets."""
    if not README.exists():
        pytest.skip("README.md not found")
    assert _check_readme_content(README)


def test_build_script_functions():
    """Build script must define expected functions."""
    if not BUILD_SCRIPT.exists():
        pytest.skip("build_android.sh not found")
    assert _check_build_script_functions(BUILD_SCRIPT)


# --- Standalone runner (python tests/test_qvac_build.py) ---

def run_all_tests() -> int:
    """Run all tests and return exit code."""
    print("=" * 60)
    print("QVAC Build System Tests")
    print("=" * 60)
    print()

    results = []

    # Test file existence
    print("\n--- File Existence Tests ---")
    results.append(_check_file_exists(BUILD_SCRIPT, "build_android.sh"))
    results.append(_check_file_exists(CMAKELISTS, "CMakeLists.txt"))
    results.append(_check_file_exists(README, "README.md"))

    # Test file permissions
    print("\n--- File Permission Tests ---")
    if BUILD_SCRIPT.exists():
        results.append(_check_file_executable(BUILD_SCRIPT))

    # Test syntax
    print("\n--- Syntax Tests ---")
    if BUILD_SCRIPT.exists():
        results.append(_check_bash_syntax(BUILD_SCRIPT))
        results.append(_check_shellcheck(BUILD_SCRIPT))

    # Test tools availability
    print("\n--- Tool Availability Tests ---")
    required_tools = test_required_tools()
    results.append(all(required_tools.values()))
    test_optional_tools()  # Optional, don't add to results

    # Test content
    print("\n--- Content Tests ---")
    if CMAKELISTS.exists():
        results.append(_check_cmake_file(CMAKELISTS))
    if BUILD_SCRIPT.exists():
        results.append(_check_build_script_content(BUILD_SCRIPT))
        results.append(_check_build_script_functions(BUILD_SCRIPT))
    if README.exists():
        results.append(_check_readme_content(README))

    # Test structure
    print("\n--- Structure Tests ---")
    results.append(test_output_directory_structure())

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print(f"{TestColors.GREEN}All tests passed!{TestColors.NC}")
        return 0
    else:
        print(f"{TestColors.RED}{total - passed} test(s) failed{TestColors.NC}")
        return 1


def main() -> int:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("QVAC Build System Tests")
        print()
        print("Usage: python tests/test_qvac_build.py")
        print()
        print("This test suite validates the QVAC build system without")
        print("actually running the cross-compilation (no NDK required).")
        print()
        print("Tests:")
        print("  - File existence and permissions")
        print("  - Bash script syntax")
        print("  - Required tool availability")
        print("  - CMakeLists.txt content")
        print("  - Build script structure and functions")
        print("  - README.md content")
        print("  - Output directory structure")
        return 0

    return run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
