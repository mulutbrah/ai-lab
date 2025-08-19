from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# --- Load .env when available (local dev) ---
local_env = os.path.join(os.path.dirname(__file__), ".env")
root_env = os.path.join(os.path.dirname(__file__), "..", ".env")

if os.path.exists(local_env):
    load_dotenv(dotenv_path=local_env, override=True)
    print(f"✅ Loaded .env from {local_env}")
elif os.path.exists(root_env):
    load_dotenv(dotenv_path=root_env, override=True)
    print(f"✅ Loaded .env from {root_env}")
else:
    print("⚠️ No local .env found, relying on Hugging Face environment variables.")

# --- Pushover helper ---
def push(text):
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")
    if not token or not user:
        print("⚠️ Pushover credentials missing, skipping push:", text)
        return
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": token, "user": user, "message": text},
    )

# --- Tool functions ---
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record a user email and details",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name"},
            "notes": {"type": "string", "description": "Additional context"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any unanswered question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unanswered question"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

# --- Main Chat Class ---
class Me:
    def __init__(self):
        # ✅ Works locally (.env) and on Spaces (Secrets)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("❌ OPENAI_API_KEY not found in env or Hugging Face secrets")
        self.openai = OpenAI(api_key=api_key)

        self.name = "muhammad lutfi ibrahim"

        # Load CV
        reader = PdfReader("data/cv.pdf")
        self.cv = "".join(page.extract_text() or "" for page in reader.pages)

        # Load Summary
        with open("data/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
            )
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly about {self.name}'s career, background, skills and experience. \
Be professional and engaging. \
If you don't know the answer, use record_unknown_question. \
If the user is engaging in discussion, steer them towards sharing their email and record it."

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## CV Profile:\n{self.cv}\n\n"
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [
            {"role": "user", "content": message}
        ]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


# --- Launch Gradio app ---
if __name__ == "__main__":
    me = Me()
    # ✅ Local: runs at localhost:7860, HuggingFace: auto-detects
    gr.ChatInterface(me.chat, type="messages").launch()
    