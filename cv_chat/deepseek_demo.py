"""
DeepSeek Integration Demo for CV Chat
This file shows how to use DeepSeek as a fallback to OpenAI
"""

import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DeepSeekClient:
    """DeepSeek API client with OpenAI-compatible interface"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    def chat_completions_create(self, model="deepseek-chat", messages=None, tools=None):
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
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        # Transform DeepSeek response to match OpenAI format
        class MockResponse:
            class MockChoice:
                class MockMessage:
                    def __init__(self, msg):
                        self.content = msg.get('content')
                        self.tool_calls = msg.get('tool_calls', [])
                
                def __init__(self, choice):
                    self.message = MockResponse.MockChoice.MockMessage(choice['message'])
                    self.finish_reason = choice.get('finish_reason')
            
            def __init__(self, result_data):
                self.choices = [MockResponse.MockChoice(result_data['choices'][0])]
        
        return MockResponse(result)

def init_ai_client():
    """Initialize AI client with automatic fallback from OpenAI to DeepSeek"""
    
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ Using OpenAI GPT-4o-mini")
        try:
            client = OpenAI(api_key=openai_key)
            # Test the API key
            test_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            closure_client = type('Client', (), {'chat': type('Chat', (), {'completions': type('Completions', (), {'create': lambda **kwargs: client.chat.completions.create(**kwargs)})()})()})()
            closure_client.model = "gpt-4o-mini"
            closure_client.client_type = "openai"
            return closure_client
        except Exception as e:
            print(f"⚠️ OpenAI API failed: {e}")
            print("🔄 Falling back to DeepSeek...")
    else:
        print("⚠️ No OpenAI API key found, using DeepSeek...")
        
    # Fallback to DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError("❌ No API keys found! Please set either OPENAI_API_KEY or DEEPSEEK_API_KEY")
        
    print("✅ Using DeepSeek")
    client = DeepSeekClient(deepseek_key)
    closure_client = type('Client', (), {'chat': type('Chat', (), {'completions': type('Completions', (), {'create': lambda **kwargs: client.chat_completions_create(**kwargs)})()})()})()
    closure_client.model = "deepseek-chat" 
    closure_client.client_type = "deepseek"
    return closure_client

# Test the integration
if __name__ == "__main__":
    try:
        client = init_ai_client()
        print(f"AI Client initialized: {client.client_type}")
        print(f"Model: {client.model}")
        
        # Test a simple conversation
        response = client.chat.completions.create(
            model=client.model,
            messages=[{"role": "user", "content": "Hello! Can you introduce yourself?"}],
            max_tokens=100
        )
        
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please set your API keys in .env file:")
        print("OPENAI_API_KEY=your_key_here")
        print("DEEPSEEK_API_KEY=your_key_here")
