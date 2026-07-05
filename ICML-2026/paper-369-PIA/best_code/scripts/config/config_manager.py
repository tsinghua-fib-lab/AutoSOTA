#!/usr/bin/env python3
"""
Configuration Manager - Helps manage local API configuration

Usage:
    python config_manager.py init          # Create local environment template
    python config_manager.py show          # Show current configuration
    python config_manager.py platforms     # Show supported platforms
    python config_manager.py check         # Check configuration completeness
"""

import sys
import os
import importlib.util

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(project_root, 'src')

# Add to sys.path for relative imports within src modules
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import modules using importlib to ensure proper loading
def import_module_from_path(module_name, file_path):
    """Import a module from a specific file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import src package and submodules
import_module_from_path('src', os.path.join(src_dir, '__init__.py'))
import_module_from_path('src.core', os.path.join(src_dir, 'core', '__init__.py'))

from src.core import (
    create_sample_env,
    print_config_summary,
    print_available_platforms,
    get_api_config,
    PLATFORM_CONFIGS,
    load_dotenv
)


def check_config():
    """Check configuration completeness"""
    print("=" * 70)
    print("🔍 Configuration Integrity Check")
    print("=" * 70)

    load_dotenv()

    issues = []
    warnings = []

    # Check each platform
    for source in PLATFORM_CONFIGS.keys():
        config = get_api_config(source)

        if config['status'] == 'missing_key':
            warnings.append(f"⚠️  {source}: API_KEY not configured")
        elif config['status'] == 'unknown_platform':
            issues.append(f"❌ {source}: Unknown platform")
        else:
            print(f"✅ {source}: Configuration normal")

    # Check runtime configuration
    from src.core import get_runtime_config
    runtime = get_runtime_config()

    if runtime['mock_mode']:
        warnings.append("⚠️  Mock mode enabled (will not call real API)")

    print("\n📊 Check Results:")
    print("-" * 70)

    if not warnings and not issues:
        print("✅ All configurations normal!")
    else:
        if issues:
            print(f"\n❌ Critical issues ({len(issues)}):")
            for issue in issues:
                print(f"  {issue}")

        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  {warning}")

    print("\n📝 Suggestions:")
    print("-" * 70)
    print("1. Run 'python config_manager.py init' to create the local configuration file")
    print("2. Edit that file to fill in your API keys")
    print("3. Run 'python config_manager.py show' to verify configuration")
    print("=" * 70)


def show_config():
    """Show current configuration"""
    print_config_summary()


def show_platforms():
    """Show supported platforms"""
    print_available_platforms()


def init_env():
    """Initialize local provider configuration"""
    create_sample_env()


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python config_manager.py init      # Create local configuration template")
        print("  python config_manager.py show      # Show configuration")
        print("  python config_manager.py platforms # Show platforms")
        print("  python config_manager.py check     # Check configuration")
        return

    command = sys.argv[1].lower()

    if command == 'init':
        init_env()
    elif command == 'show':
        show_config()
    elif command == 'platforms':
        show_platforms()
    elif command == 'check':
        check_config()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: init, show, platforms, check")


if __name__ == "__main__":
    main()
