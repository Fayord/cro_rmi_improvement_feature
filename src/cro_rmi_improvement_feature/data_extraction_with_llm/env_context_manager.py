"""
Simple Environment Context Demo
Shows the key concepts you asked about:
1. Set many env vars with prefixes
2. Use context wrapper with same env name
3. Show isolation - outside context doesn't affect original
"""

import os
from contextlib import contextmanager


class EnvContextManager:
    """Context manager for temporarily changing environment variables"""

    def __init__(self, env_vars):
        self.env_vars = env_vars
        self.original_values = {}

    def __enter__(self):
        """Set environment variables and store original values"""
        for key, value in self.env_vars.items():
            self.original_values[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original environment variables"""
        for key, original_value in self.original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def print_env(name):
    """Print a single environment variable"""
    value = os.environ.get(name, "NOT SET")
    print(f"  {name} = {value}")


def main():
    print("🎯 Simple Environment Context Demo")
    print("=" * 50)

    # 1. Set many env vars with prefixes
    print("\n1️⃣ Setting many environment variables with prefixes:")

    # Set up different configurations with prefixes
    configs = {
        "config-a": {
            "API_KEY": "key_a_123",
            "MODEL_NAME": "model_a",
            "TEMPERATURE": "0.1",
        },
        "config-b": {
            "API_KEY": "key_b_456",
            "MODEL_NAME": "model_b",
            "TEMPERATURE": "0.2",
        },
        "config-c": {
            "API_KEY": "key_c_789",
            "MODEL_NAME": "model_c",
            "TEMPERATURE": "0.3",
        },
    }
    # set default env vars
    os.environ["API_KEY"] = "default_key"
    os.environ["MODEL_NAME"] = "default_model"
    os.environ["TEMPERATURE"] = "0.5"

    print("Available configurations:")
    for name, env_vars in configs.items():
        print(f"  {name}: {env_vars}")

    # 2. Show original environment
    print("\n2️⃣ Original environment:")
    print_env("API_KEY")
    print_env("MODEL_NAME")
    print_env("TEMPERATURE")

    # 3. Use context wrapper with same env name
    print("\n3️⃣ Using context wrapper with same environment names:")

    for config_name, env_vars in configs.items():
        print(f"\n🔄 Testing {config_name}:")

        # Inside context - should have different values
        with EnvContextManager(env_vars):
            print(f"  📋 Inside {config_name} context:")
            print_env("API_KEY")
            print_env("MODEL_NAME")
            print_env("TEMPERATURE")

        # Outside context - should be back to original
        print(f"  📋 Outside {config_name} context:")
        print_env("API_KEY")
        print_env("MODEL_NAME")
        print_env("TEMPERATURE")

    # 4. Show isolation - outside context doesn't affect original
    print("\n4️⃣ Final environment (should be unchanged):")
    print_env("API_KEY")
    print_env("MODEL_NAME")
    print_env("TEMPERATURE")

    print("\n✅ Demo completed! Environment variables are isolated within contexts.")


if __name__ == "__main__":
    main()
