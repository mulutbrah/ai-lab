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
    push(f"📩 Recruiter info: {name} ({email}) | {notes}")
    return {"✅ Recruiter recorded": f"{name} ({email})"}

def record_unknown_question(question):
    push(f"🤔 Unknown recruiter question: {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Record recruiter contact details (email, name, notes).",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "Recruiter's email address"},
            "name": {"type": "string", "description": "Recruiter's name"},
            "notes": {"type": "string", "description": "Additional context or position details"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any recruiter question not answered by CV/summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Unanswered recruiter question"},
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
            print(f"⚙️ Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
            )
        return results

    def system_prompt(self):
        system_prompt = f"""
You are acting as {self.name}, a professional software engineer.
You are answering questions from recruiters on {self.name}'s website.

🎯 Goals:
- Present {self.name}'s career, technical skills, and experience clearly.
- Be professional, concise, and engaging.
- If a recruiter shares contact info, record it with `record_user_details`.
- If you don’t know an answer, use `record_unknown_question`.

📌 Context for recruiter:
## Summary:
{self.summary}

## CV Profile:
{self.cv}
"""
        return system_prompt.strip()

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

# --- Gradio Recruiter Info Form ---
def recruiter_form(name, email, notes):
    return record_user_details(email=email, name=name, notes=notes)

# --- Launch Gradio app ---
if __name__ == "__main__":
    me = Me()

    chat = gr.ChatInterface(
        fn=me.chat,
        type="messages",
        title="💼 Chat with Muhammad Lutfi Ibrahim",
        description="Ask me about my career, technical skills, and experience.",
        examples=[
            ["Can you tell me about your background?"],
            ["What technical skills are you strongest in?"],
            ["Are you open to remote work?"],
        ],
    )

    form = gr.Interface(
        fn=recruiter_form,
        inputs=[
            gr.Textbox(label="Your Name"),
            gr.Textbox(label="Your Email"),
            gr.Textbox(label="Position / Notes (optional)"),
        ],
        outputs="json",
        title="📩 Share your details",
        description="Leave your contact info if you'd like me to follow up.",
    )

    # Tab layout: Chat + Lead Form
    demo = gr.TabbedInterface([chat, form], ["🤖 Chat", "📩 Leave Info"])
    demo.launch()
