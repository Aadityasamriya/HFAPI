# 🚅 Railway.com Deployment Guide - Hugging Face By AadityaLabs AI

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)  
- [Step-by-Step Deployment](#-step-by-step-deployment)
- [Environment Variables Setup](#-environment-variables-setup)
- [Database Configuration](#-database-configuration)
- [Post-Deployment Testing](#-post-deployment-testing)
- [Troubleshooting](#-troubleshooting)
- [Performance & Scaling](#-performance--scaling)

---

## 🚀 Quick Start

Deploy your AI bot to Railway.com in under 5 minutes:

### Method 1: One-Click Deploy (Recommended)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/huggingface-ai-bot)

⚡ **True One-Click Deployment**: This template automatically:
- ✅ Forks the repository to your GitHub account
- ✅ Configures Python 3.11 build environment
- ✅ Sets up MongoDB connection prompts
- ✅ Guides you through required environment variables
- ✅ Deploys instantly with proper configuration

### Method 2: GitHub Integration
1. Fork this repository
2. Visit [railway.app](https://railway.app) and create account
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your forked repository
5. Add required environment variables
6. Deploy automatically

---

## ✅ Prerequisites

Before deploying, ensure you have:

- [ ] **Railway.com account** (free tier available)
- [ ] **Telegram bot token** from @BotFather
- [ ] **MongoDB database** (Atlas free tier or Railway add-on)
- [ ] **Basic understanding** of environment variables

**Estimated setup time:** 10-15 minutes for first-time users

---

## 🎯 Step-by-Step Deployment

### Step 1: Create Railway Project

1. **Login to Railway**
   - Visit [railway.app](https://railway.app)
   - Sign up/login with GitHub (recommended) or email

2. **Create New Project**
   - Click "New Project" button
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your repositories

3. **Select Repository**
   - Choose your forked repository
   - Railway will automatically detect it's a Python project
   - Click "Deploy Now"

### Step 2: Configure Build Settings

Railway will automatically:
- ✅ Detect `nixpacks.toml` for build configuration
- ✅ Install Python 3.11 and dependencies
- ✅ Use `python main.py` as start command
- ✅ Set up worker process (not web service)

**No manual build configuration needed!**

### Step 3: Add Environment Variables

In Railway Dashboard → Your Project → Variables tab:

#### Required Variables (CRITICAL):
```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority
```

#### Security Variables (REQUIRED for Production):
```bash
ENCRYPTION_SEED=your_32_character_encryption_key_here_secure_random_string
```
⚠️ **CRITICAL**: Without a persistent `ENCRYPTION_SEED`, users will lose access to stored API keys after restarts.

#### Optional Variables:
```bash
OWNER_ID=123456789
BOT_NAME=My AI Assistant Pro
```

⚠️ **Important:** Click "Save" after adding each variable

### Step 4: Deploy & Monitor

1. **Trigger Deployment**
   - Railway auto-deploys after adding variables
   - Or click "Deploy" button manually
   - Monitor build logs in real-time

2. **Check Deployment Status**
   - Green ✅ = Successful deployment
   - Red ❌ = Build/runtime error (check logs)
   - Building 🔄 = Deployment in progress

---

## 🔧 Environment Variables Setup

### Core Configuration

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | ✅ Yes | Bot token from @BotFather | `1234567890:ABC...` |
| `MONGODB_URI` | ✅ Yes | MongoDB connection string | `mongodb+srv://user:pass@...` |
| `ENCRYPTION_SEED` | 🚨 **REQUIRED** | 32+ character encryption key | **MUST be set for production** |
| `OWNER_ID` | ❌ Optional | Your Telegram user ID for admin | `123456789` |

### Advanced Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUEST_TIMEOUT` | `30` | API request timeout (seconds) |
| `MAX_RETRIES` | `3` | Maximum retry attempts |
| `MAX_CHAT_HISTORY` | `20` | Messages to keep in context |
| `MAX_RESPONSE_LENGTH` | `4000` | Maximum response length |

### How to Set Variables in Railway:

1. **Navigate to Project**
   - Go to Railway dashboard
   - Select your deployed project

2. **Access Variables Tab**
   - Click "Variables" in the sidebar
   - You'll see existing variables if any

3. **Generate Secure ENCRYPTION_SEED** (CRITICAL STEP):
   ```bash
   # Run this command to generate a secure seed:
   python3 -c "import secrets, base64; print('ENCRYPTION_SEED=' + base64.b64encode(secrets.token_bytes(32)).decode())"
   ```
   - Copy the entire output and add as an environment variable
   - **WARNING**: Save this value securely - losing it means users lose API key access

4. **Add New Variables**
   - Click "+ New Variable" button
   - Enter variable name and value
   - Click "Save" - deployment will restart automatically

5. **Edit Existing Variables**
   - Click on any variable to edit
   - Make changes and save
   - Service restarts automatically

---

## 🗄️ Database Configuration

### Option 1: MongoDB Atlas (Recommended - Free)

1. **Create MongoDB Atlas Account**
   - Visit [cloud.mongodb.com](https://cloud.mongodb.com)
   - Sign up for free account
   - Create new cluster (M0 Sandbox - Free)

2. **Database Security Setup**
   - **Database Access**: Create user with `readWrite` permissions
   - **Network Access**: Add IP `0.0.0.0/0` (allow all)
   - **Note**: In production, restrict to Railway's IP ranges

3. **Get Connection String**
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy connection string (mongodb+srv format)
   - Replace `<username>` and `<password>` with your credentials
   - Use this as `MONGODB_URI` in Railway

4. **Verify Connection**
   - Test connection string format: `mongodb+srv://username:password@cluster.mongodb.net/botdb?retryWrites=true&w=majority`
   - Ensure it includes database name and required parameters

### Option 2: Railway MongoDB Add-on

1. **Add MongoDB Service**
   - In Railway project dashboard
   - Click "+ New" → "Add Service" 
   - Select "MongoDB" from templates
   - Deploy MongoDB service

2. **Get Connection Details**
   - Railway provides these variables automatically:
     - `MONGO_URL`
     - `MONGOHOST`
     - `MONGOPORT`
     - `MONGOUSER`
     - `MONGOPASSWORD`

3. **Configure Connection**
   - Use `MONGO_URL` as your `MONGODB_URI`
   - Or construct from individual variables
   - Internal Railway services connect automatically

### Database Best Practices

- ✅ **Use TLS**: Ensure connection string includes TLS settings
- ✅ **Strong Password**: Use complex passwords for database users  
- ✅ **Regular Backups**: Enable automated backups in Atlas
- ✅ **Monitor Usage**: Track database storage and requests
- ⚠️ **IP Restrictions**: In production, limit network access

---

## ✅ Post-Deployment Testing

### Automated Health Checks

After deployment, Railway automatically monitors:
- ✅ **Process Status**: Bot process running without crashes
- ✅ **Memory Usage**: Monitor RAM consumption
- ✅ **Network Activity**: Database connections and API calls
- ✅ **Error Rates**: Exception tracking in logs

### Manual Testing Checklist

#### 1. Verify Bot Startup (2 minutes)
```bash
# Check Railway logs for these success messages:
✅ "Configuration validated successfully" 
✅ "Bot authenticated: @YourBotName (ID: 123456789)"
✅ "Storage backend connected successfully"
✅ "Bot is now LIVE and listening for messages"
```

#### 2. Test Basic Commands (3 minutes)
- [ ] Send `/start` to your bot
- [ ] Verify welcome message appears with setup instructions
- [ ] Test `/settings` command to access configuration
- [ ] Try `/settings` to access configuration menu

#### 3. Test AI Functionality (5 minutes)
- [ ] Set up Hugging Face API key via bot settings
- [ ] Send simple text message: "Hello, how are you?"
- [ ] Try code request: "Write a Python function to calculate fibonacci"
- [ ] Test image generation: "Generate an image of a sunset"
- [ ] Verify responses are contextual and intelligent

#### 4. Test Advanced Features (Optional)
- [ ] Upload a PDF file for text extraction
- [ ] Upload an image for OCR analysis  
- [ ] Test `/newchat` to clear conversation history
- [ ] Try complex multi-turn conversation

### Performance Verification

Monitor in Railway dashboard:
- **Response Times**: < 10 seconds for most requests
- **Memory Usage**: < 256MB typical usage
- **Error Rate**: < 5% for normal operation
- **Database Connections**: Stable connection pool

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue 1: Bot Not Starting
```
❌ Error: Bot crashes immediately after deployment
```
**Solutions:**
1. **Check Environment Variables**
   - Verify `TELEGRAM_BOT_TOKEN` format: `numbers:letters_and_symbols`
   - Ensure `MONGODB_URI` includes username, password, and database name
   - Variables should not have quotes around values in Railway

2. **Check Railway Logs**
   ```bash
   # Look for specific error messages in deployment logs:
   - "TELEGRAM_BOT_TOKEN environment variable is missing"
   - "Database configuration is required"
   - "Bot authentication failed"
   ```

3. **Verify Bot Token**
   - Message @BotFather on Telegram
   - Send `/mybots` to list your bots
   - Click on your bot → "API Token" to verify

#### Issue 2: Database Connection Failed
```
❌ Error: "Database configuration is required" or connection timeouts
```
**Solutions:**
1. **MongoDB Atlas Issues**
   - Check Network Access allows `0.0.0.0/0`
   - Verify username/password in connection string
   - Ensure cluster is not paused (Atlas pauses after inactivity)

2. **Connection String Format**
   ```bash
   # Correct format:
   mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority
   
   # Common mistakes:
   - Missing database name after .net/
   - Special characters in password not URL-encoded
   - Using mongodb:// instead of mongodb+srv://
   ```

3. **Railway MongoDB Add-on Issues**
   - Ensure both services are in same project
   - Check if `MONGO_URL` variable exists
   - Verify internal networking is enabled

#### Issue 3: API Key Issues
```
❌ Error: Users can't use AI features or get "API key invalid" errors
```
**Solutions:**
1. **User Education**
   - Users need their own Hugging Face API key
   - Guide them to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - API key must have "Read" permissions minimum

2. **Bot Setup Issues**
   - Try `/start` command to reset user configuration
   - Use `/settings` to update API key
   - Test with a different Hugging Face account

#### Issue 4: High Memory Usage
```
⚠️ Warning: Memory usage > 500MB or frequent restarts
```
**Solutions:**
1. **Optimize Configuration**
   - Reduce `MAX_CHAT_HISTORY` to `10-15`
   - Lower `MAX_RESPONSE_LENGTH` to `2000`
   - Enable `USE_SMART_CACHING=true`

2. **Scale Railway Plan**
   - Upgrade to Hobby plan ($5/month) for 1GB RAM
   - Monitor usage in Railway metrics dashboard

#### Issue 5: Slow Response Times
```
⚠️ Warning: Bot responses taking > 30 seconds
```
**Solutions:**
1. **Model Optimization**
   - Bot automatically uses fallback models
   - Hugging Face API can be slow during peak times
   - Consider upgrading to Hugging Face Pro for faster inference

2. **Database Optimization**
   - Ensure MongoDB Atlas cluster is not paused
   - Use database in same region as Railway deployment
   - Consider upgrading MongoDB Atlas tier

### Getting Help

#### Railway-Specific Help
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Status**: [status.railway.app](https://status.railway.app)

#### Bot-Specific Help
- **Check Logs**: Railway Dashboard → Your Project → Logs tab
- **Debug Mode**: Add `LOG_LEVEL=DEBUG` environment variable
- **Health Check**: Monitor system messages in Railway logs

#### Database Help
- **MongoDB Atlas Support**: [support.mongodb.com](https://support.mongodb.com)
- **Connection String Builder**: Use Atlas connection guide
- **Network Debug**: Verify IP whitelist includes `0.0.0.0/0`

---

## 📈 Performance & Scaling

### Current Capacity (Free Railway Tier)
- **Concurrent Users**: 20-50 users simultaneously
- **Monthly Usage**: ~500MB memory, minimal CPU
- **Database**: MongoDB Atlas M0 (512MB storage, 100 connections)
- **Response Time**: 3-15 seconds (depends on Hugging Face API)

### Scaling Options

#### Railway Plan Upgrades
| Plan | Price | RAM | Features |
|------|-------|-----|----------|
| **Free** | $0 | 512MB | 1 project, 500 hours/month |
| **Hobby** | $5/month | 1GB | 3 projects, unlimited hours |
| **Pro** | $20/month | 8GB | 10 projects, priority support |

#### MongoDB Atlas Scaling
| Tier | Price | Storage | Connections |
|------|-------|---------|-------------|
| **M0 (Free)** | $0 | 512MB | 100 |
| **M2** | $9/month | 2GB | 500 |
| **M5** | $25/month | 5GB | 1,500 |

### Performance Optimization

#### Application Level
1. **Enable Smart Caching**
   ```bash
   # Add to Railway environment variables:
   USE_SMART_CACHING=true
   ENABLE_MODEL_FALLBACKS=true
   ```

2. **Optimize Chat History**
   ```bash
   # Reduce memory usage:
   MAX_CHAT_HISTORY=10
   MAX_RESPONSE_LENGTH=2000
   ```

3. **Connection Pooling**
   - MongoDB connections are automatically pooled
   - Railway handles connection management
   - No additional configuration needed

#### Infrastructure Level
1. **Regional Optimization**
   - Deploy Railway project in US-West (default)
   - Use MongoDB Atlas in same region
   - Consider CDN for static assets (if any)

2. **Monitoring Setup**
   - Enable Railway metrics dashboard
   - Set up alerts for high memory usage
   - Monitor database connection counts

### Traffic Handling

#### Expected Load Patterns
- **Peak Hours**: 6 PM - 11 PM local time
- **Usage Spikes**: During model updates or feature releases
- **Concurrent Sessions**: 3-5 requests per user per session

#### Auto-Scaling Features
- **Railway**: Automatically restarts on memory limits
- **MongoDB Atlas**: Auto-scales connections
- **Bot Logic**: Built-in rate limiting and error handling

---

## 🎉 Deployment Complete!

### Success Indicators
- ✅ Bot appears online in Telegram
- ✅ `/start` command works with welcome message
- ✅ Users can set Hugging Face API keys
- ✅ AI responses generate successfully
- ✅ Database connections stable
- ✅ Railway shows healthy deployment status

### Next Steps
1. **Share Your Bot**: Send bot link to users for testing
2. **Monitor Logs**: Check Railway logs for any issues
3. **User Onboarding**: Guide users through `/start` setup process
4. **Scale Planning**: Monitor usage and upgrade tiers as needed

### Congratulations! 🎊
Your **Hugging Face By AadityaLabs AI** bot is now deployed on Railway.com and ready to provide cutting-edge AI capabilities to your users!

---

**Railway Deployment Guide Version 1.0**  
*Last Updated: September 18, 2025*