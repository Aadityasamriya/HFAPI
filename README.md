# 🤖 AI Assistant Pro - Advanced Telegram Bot

A sophisticated Telegram bot powered by **2024-2025's latest AI models** including Llama-3.2, Qwen2.5, StarCoder2-15B, and FLUX.1. Superior to ChatGPT, Grok, and Gemini with intelligent model routing and multi-modal capabilities.

## ⚡ Quick Railway Deployment

### 1. One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/deploy)

### 2. GitHub Deploy
1. Fork this repository
2. Connect to Railway: [railway.app/new](https://railway.app/new)
3. Select "Deploy from GitHub repo" 
4. Choose your forked repository
5. Add environment variables (see below)
6. Deploy automatically triggers

## 🔧 Required Environment Variables

Set these in Railway Dashboard → Variables:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
MONGO_URI=your_mongodb_connection_string
```

### Optional Variables:
```bash
OWNER_ID=your_telegram_user_id_for_admin_features
ENCRYPTION_KEY=your_fernet_encryption_key_for_api_storage
```

## 🚀 Getting Started

### 1. Create Telegram Bot
- Message [@BotFather](https://t.me/BotFather)
- Use `/newbot` command
- Copy your bot token

### 2. Get MongoDB Database
- Create free MongoDB Atlas cluster
- Get connection string (mongodb+srv format)

### 3. Deploy on Railway
- Use GitHub integration for auto-deployment
- Set environment variables
- Bot starts automatically

## 🎯 Features

### Latest AI Models (2024-2025)
- **🧠 Text**: Llama-3.2-3B, Qwen2.5-7B (29+ languages)
- **💻 Code**: StarCoder2-15B (80+ programming languages) 
- **🎨 Images**: FLUX.1-schnell (state-of-the-art generation)
- **📊 Analysis**: Advanced sentiment & emotion detection

### Smart Capabilities
- ✨ Intelligent model routing
- 🔄 Context-aware conversations  
- 🛡️ Enterprise-grade security
- 📱 Rich UI with inline keyboards
- ⚡ Multi-modal AI interactions

## 💬 User Commands

- `/start` - Welcome & API key setup
- `/newchat` - Clear conversation history
- `/settings` - Manage preferences
- `/help` - Comprehensive help guide

## 🛡️ Security Features

- 🔐 Encrypted API key storage
- 🛡️ TLS-enforced database connections
- 🔒 Markdown injection protection
- ⚡ Rate limiting (15 requests/minute)
- 🚫 Comprehensive error handling

## 📋 User Setup (After Deployment)

1. **Get Hugging Face API Key**
   - Visit: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create token with "Read" permissions
   - Copy the token (starts with `hf_`)

2. **Start Using Bot**
   - Send `/start` to your bot
   - Follow setup instructions
   - Provide your Hugging Face API key
   - Start chatting with advanced AI!

## 🔧 Development

### Local Setup
```bash
git clone <your-repo-url>
cd ai-assistant-pro
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your values
python main.py
```

### Project Structure
```
├── main.py              # Bot entry point
├── requirements.txt     # Dependencies
├── railway.json        # Railway configuration
├── nixpacks.toml       # Build configuration
├── bot/
│   ├── config.py       # Environment configuration
│   ├── database.py     # MongoDB integration
│   ├── security_utils.py # Security utilities
│   ├── core/
│   │   ├── router.py   # Intelligent routing
│   │   └── model_caller.py # AI model integration
│   └── handlers/
│       ├── command_handlers.py # Command handling
│       └── message_handlers.py # Message processing
```

## ⚖️ License

This project is open source. Users provide their own Hugging Face API keys for AI model access.

## 🎉 Why Choose This Bot?

- **Latest Models**: Access to 2024-2025's most advanced AI
- **Free to Run**: Only pay for Railway hosting (~$5/month)
- **No Limits**: Use your own Hugging Face API quotas
- **Superior Intelligence**: Outperforms ChatGPT with newer models
- **Multi-Modal**: Text, code, images, analysis - all in one bot
- **Production Ready**: Enterprise security and reliability

Deploy now and give your users access to the world's most advanced AI models!