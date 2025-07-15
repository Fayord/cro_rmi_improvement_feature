from pyzerox import zerox
from env_context_manager import EnvContextManager
import os
import asyncio


async def demo_run_zerox():
    """Demonstrate running zerox with environment variables"""
    print("\n" + "=" * 60)
    print("🌍 ZEROX DEMO")
    print("=" * 60)

    # get env vars
    env_vars = {
        "AZURE_API_KEY": os.getenv("AZURE_API_KEY"),
        "AZURE_API_BASE": os.getenv("AZURE_API_BASE"),
        "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
    }
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # run zerox with context manager
    with EnvContextManager(env_vars):
        result = await zerox(
            file_path=f"{dir_path}/วิธีการสมัครเป็น IB PC.V.pdf",
            model="azure/gpt-4o-mini",
        )

    print(result)


if __name__ == "__main__":
    asyncio.run(demo_run_zerox())
