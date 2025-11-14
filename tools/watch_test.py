import asyncio
from gz_helpers.collision_watch import watch_contacts

async def main():
    q = asyncio.Queue()
    await watch_contacts(q)

if __name__ == "__main__":
    asyncio.run(main())
