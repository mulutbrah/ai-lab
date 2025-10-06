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
def record_user_details(email, name, notes=""):
    record = {
        "name": name,
        "email": email,
        "notes": notes,
    }

    with open("recruiters.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    push(f"📩 New recruiter lead: {name} ({email})\nNotes: {notes}")

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

# --- DeepSeek Integration ---
class DeepSeekClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Create a chat.completions.create method that matches OpenAI
        chat = type('Chat', (), {
            'completions': type('Completions', (), {
                'create': self._make_request
            })()
        })()
        
        # Attach chat to this instance
        self.chat = chat
    
    def _make_request(self, model="deepseek-chat", messages=None, tools=None, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }
        
        print(f"🔍 DeepSeek Request: {data}")
        response = requests.post(self.base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            print(f"❌ DeepSeek Error {response.status_code}: {response.text}")
            response.raise_for_status()
        
        result = response.json()
        
        # Transform DeepSeek response to match OpenAI format
        class Response:
            def __init__(self, result_data):
                if result_data.get('choices'):
                    choice = result_data['choices'][0]
                    self.choices = [Choice(choice)]
                else:
                    self.choices = []
        
        class Choice:
            class Message:
                def __init__(self, msg_data):
                    self.content = msg_data.get('content')
                    self.tool_calls = msg_data.get('tool_calls', [])
                    if msg_data.get('tool_calls'):
                        # Format tool_calls to match OpenAI format
                        self.tool_calls = [
                            type('ToolCall', (), {
                                'id': tc.get('id'),
                                'function': type('Function', (), {
                                    'name': tc.get('function', {}).get('name'),
                                    'arguments': json.dumps(tc.get('function', {}).get('arguments', {}))
                                })()
                            })() for tc in msg_data.get('tool_calls', [])
                        ]
            
            def __init__(self, choice_data):
                self.message = self.Message(choice_data['message'])
                self.finish_reason = choice_data.get('finish_reason')
        
        return Response(result)

# --- Ollama Integration (Local, Free) ---
class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.model = "llama3"
        
        # Create a chat.completions.create method that matches OpenAI
        chat = type('Chat', (), {
            'completions': type('Completions', (), {
                'create': self._make_request
            })()
        })()
        
        self.chat = chat
    
    def test_connection(self):
        """Test if Ollama is running"""
        response = requests.get(f"{self.base_url}/tags")
        if response.status_code != 200:
            raise Exception("Ollama not running. Install with: brew install ollama && ollama run llama3")
    
    def _make_request(self, model=None, messages=None, tools=None, **kwargs):
        # Convert OpenAI format to Ollama format
        prompt = self._messages_to_prompt(messages)
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(f"{self.base_url}/generate", json=data)
        response.raise_for_status()
        
        result = response.json()
        
        # Transform Ollama response to match OpenAI format
        class Response:
            class Choice:
                class Message:
                    def __init__(self, content):
                        self.content = content
                        self.tool_calls = []  # Ollama doesn't support function calling yet
                
                def __init__(self, content):
                    self.message = Response.Choice.Message(content)
                    self.finish_reason = "stop"
            
            def __init__(self, content):
                self.choices = [Response.Choice(content)]
        
        return Response(result.get('response', ''))
    
    def _messages_to_prompt(self, messages):
        """Convert OpenAI messages format to Ollama prompt"""
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt

# --- Main Chat Class ---
class Me:
    def __init__(self):
        self.name = "muhammad lutfi ibrahim"
        
        # Initialize AI client with fallback
        self.ai_client = self._init_ai_client()
        
        # Load CV
        reader = PdfReader("data/cv.pdf")
        self.cv = "".join(page.extract_text() or "" for page in reader.pages)

        # Load Summary
        with open("data/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
            
    def _init_ai_client(self):
        # Try OpenAI first
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print("✅ Using OpenAI GPT-4o-mini")
            try:
                client = OpenAI(api_key=openai_key)
                # Test the API key by making a simple request
                test_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1
                )
                return client
            except Exception as e:
                print(f"⚠️ OpenAI API failed: {e}")
                print("🔄 Falling back to DeepSeek...")
        else:
            print("⚠️ No OpenAI API key found, using DeepSeek...")
            
        # Fallback to DeepSeek
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            print("✅ Using DeepSeek")
            return DeepSeekClient(deepseek_key)
        
        # Final fallback to Ollama (completely free, local)
        print("🔄 Trying Ollama (local, free)...")
        try:
            ollama_client = OllamaClient()
            ollama_client.test_connection()
            print("✅ Using Ollama (local)")
            return ollama_client
        except Exception as e:
            print(f"⚠️ Ollama not available: {e}")
        
        print("❌ No AI service available! Please:")
        print("   1. Set OPENAI_API_KEY, or")
        print("   2. Set DEEPSEEK_API_KEY with credits, or") 
        print("   3. Install Ollama: brew install ollama && ollama run llama3")
        raise RuntimeError("❌ No AI service available")

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
            # Use appropriate model based on client type
            if isinstance(self.ai_client, OpenAI):
                response = self.ai_client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages, tools=tools
                )
            elif hasattr(self.ai_client, 'model') and self.ai_client.model == "llama3":
                # Ollama client - no function calling support yet
                response = self.ai_client.chat.completions.create(
                    messages=messages
                )
            else:
                # DeepSeek client already handles model internally
                response = self.ai_client.chat.completions.create(
                    messages=messages, tools=tools
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
    record_user_details(email=email, name=name, notes=notes)
    return f"✅ Thanks {name}, your details have been recorded! I'll follow up with you soon."

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
        outputs="text",
        title="📩 Share your details",
        description="Leave your contact info if you'd like me to follow up.",
    )

    # Tab layout: Chat + Lead Form
    demo = gr.TabbedInterface([chat, form], ["🤖 Chat", "📩 Leave Info"])
    demo.launch()
