"""Terminal chat client for the librarian API: run `make serve` in another tab, then `make chat`."""

import httpx2

CHAT_URL = 'http://localhost:8000/chat'


def main() -> None:
    history: list[dict] = []
    print('Ask-a-Librarian — empty line or Ctrl-C to exit')
    while True:
        try:
            message = input('\nyou > ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            break
        try:
            response = httpx2.post(CHAT_URL, json={'message': message, 'history': history}, timeout=120)
        except httpx2.ConnectError:
            print('cannot reach the API — is `make serve` running?')
            continue
        except httpx2.HTTPError as e:
            print(f'request failed: {type(e).__name__}: {e}')
            continue
        if response.status_code != 200:
            print(f'API error {response.status_code}: {response.text[:300]}')
            continue
        body = response.json()
        turn = body['history'][len(history) + 1 :]
        tools = [c['function']['name'] for m in turn for c in m.get('tool_calls') or []]
        if tools:
            print(f'[tools: {", ".join(tools)}]')
        print(f'\nlibrarian > {body["reply"]}')
        history = body['history']


if __name__ == '__main__':
    main()
