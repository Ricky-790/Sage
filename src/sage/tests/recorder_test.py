import asyncio

from sage.recorder import Recorder

recorder = Recorder("src/sage/tests/outputs/recorder_test/traces.jsonl")


@recorder.observe(kind="agent")
async def run_agent(query):
    result = await search_tool(query)
    return f"Search result: {result}"


@recorder.observe(kind="tool")
async def search_tool(query):
    return {"matches": [query]}


async def main():
    result = await run_agent("observability")


asyncio.run(main())
