# AI Lab

Mastering AI through my daily projects

## Projects

### CV Chat Bot 🤖

An intelligent AI chatbot that acts as Muhammad Lutfi Ibrahim's digital representative, capable of answering questions about his career, background, skills, and experience. The bot uses OpenAI's GPT-4o-mini model with function calling capabilities to interact with users and collect contact information.

#### Features

- **AI-Powered Chat Interface**: Built with Gradio for an intuitive web-based chat experience
- **CV Integration**: Automatically loads and processes CV from PDF format
- **Smart Tool Calling**: Uses AI function calling to:
  - Record user contact details when they're interested in connecting
  - Track unanswered questions for continuous improvement
- **AI Provider Fallback**: Automatically falls back from OpenAI to DeepSeek if needed
- **Pushover Notifications**: Sends real-time notifications about user interactions
- **Professional Representation**: Maintains professional tone while engaging potential clients/employers

#### Tech Stack

- **Backend**: Python with OpenAI API
- **Frontend**: Gradio web interface
- **AI Models**: GPT-4o-mini (OpenAI) with automatic fallback to DeepSeek
- **Function Calling**: Supported by both OpenAI and DeepSeek
- **PDF Processing**: PyPDF for CV extraction
- **Notifications**: Pushover API integration
- **Environment**: Python-dotenv for configuration

#### How to Run

1. **Install Dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the project root with:
   ```env
   # Primary: OpenAI API Key
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Fallback: DeepSeek API Key (FREE alternative to OpenAI!)
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # Optional: Pushover notifications
   PUSHOVER_TOKEN=your_pushover_token
   PUSHOVER_USER=your_pushover_user
   ```
   
   **🔥 You only need ONE of OpenAI or DeepSeek keys!** The app automatically falls back to DeepSeek if OpenAI fails.

3. **Run the Application**:
   ```bash
   # Option 1: Run the Python script
   cd cv_chat
   python app.py
   
   # Option 2: Run the Jupyter notebook
   jupyter notebook main.ipynb
   ```

4. **Access the Interface**:
   - Local: http://127.0.0.1:7860
   - The app will automatically launch in your browser

#### Project Structure

```
cv_chat/
├── app.py              # Main application script
├── main.ipynb          # Jupyter notebook version
├── requirements.txt    # Project dependencies
├── README.md          # Hugging Face Spaces config
└── data/
    ├── cv.pdf         # CV document
    └── summary.txt    # Professional summary
```

#### Key Components

- **Me Class**: Core chatbot logic with OpenAI integration
- **Tool Functions**: `record_user_details()` and `record_unknown_question()`
- **System Prompt**: Comprehensive prompt that includes CV and summary context
- **Gradio Interface**: Clean, responsive chat UI

#### Deployment

The project is configured for deployment on Hugging Face Spaces with automatic environment variable detection from HF Secrets.

---

*This project demonstrates practical AI application development, combining modern LLM capabilities with real-world business use cases.*
