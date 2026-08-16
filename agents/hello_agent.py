"""A first real AGENT: an LLM that can call tools, using PydanticAI + Claude.

An "agent" = a model given tools and a loop, so it can decide to act (call a
function), see the result, and continue until it can answer. This one has a
weather tool and a dice tool; watch which it chooses.

    # needs ANTHROPIC_API_KEY in .env
    uv run python agents/hello_agent.py
"""

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

load_dotenv()

agent = Agent(
    "anthropic:claude-sonnet-5",
    system_prompt="You are a helpful assistant. Use tools when relevant.",
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Return the current weather for a city (stubbed demo data)."""
    fake = {"tokyo": "18C, rain", "london": "11C, cloudy", "austin": "34C, sunny"}
    return fake.get(city.lower(), "unknown city")


@agent.tool
def roll_dice(ctx: RunContext[None], sides: int = 6) -> int:
    """Roll an n-sided die."""
    import random

    return random.randint(1, sides)


if __name__ == "__main__":
    result = agent.run_sync(
        "What's the weather in Tokyo, and roll me a 20-sided die."
    )
    print(result.output)
