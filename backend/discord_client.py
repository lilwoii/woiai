import os, httpx
from dotenv import load_dotenv
load_dotenv()

async def send_discord(message: str, webhook_url: str | None = None):
    url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
    if not url:
        return {'ok': True, 'skipped': True}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={'content': message})
    except Exception:
        pass
    return {'ok': True}
