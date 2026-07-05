import os
import sys
import time

from google import genai

import dotenv
dotenv.load_dotenv()


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not api_key:
        print("GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key, project=project)

    for job in client.batches.list():
        name = getattr(job, "name", "")
        state = getattr(job, "state", None)
        display = getattr(job, "display_name", "")
        created = getattr(job, "create_time", "")
        updated = getattr(job, "update_time", "")
        print(type(state))
        print(f"{name} | state={getattr(state, 'name', state)} | display={display} | created={created} | updated={updated}")


if __name__ == "__main__":
    main()
