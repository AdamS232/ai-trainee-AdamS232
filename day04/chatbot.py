import requests, json, sys

HOST = "http://localhost:11434"

def list_models():
    r = requests.get(f"{HOST}/api/tags").json()
    return [m["name"] for m in r.get("models", [])]

def chat_stream(model, messages):
    with requests.post(f"{HOST}/api/chat",
                       json={"model": model, "messages": messages, "stream": True},
                       stream=True) as r:
        full = ""
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            sys.stdout.write(piece); sys.stdout.flush()
            full += piece
            if chunk.get("done"):
                print()
                return full

def main():
    print("Available models:", list_models())
    model = input("Model (e.g., llama3.2:3b): ").strip() or "llama3.2:3b"
    system = input("System prompt (Enter for default): ").strip() \
             or "You are a helpful, concise assistant."
    history = [{"role": "system", "content": system}]
    print(f"\n--- Chatting with {model}. Commands: /clear /exit /swap <model> ---")
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/exit": break
        if user == "/clear":
            history = [{"role": "system", "content": system}]; continue
        if user.startswith("/swap "):
            model = user.split(" ", 1)[1]
            print(f"Swapped to {model}."); continue
        history.append({"role": "user", "content": user})
        print("Assistant: ", end="")
        reply = chat_stream(model, history)
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()