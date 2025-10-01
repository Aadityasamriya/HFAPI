# Railway.com Deployment Guide - Telegram Bot

## ✅ Fixed Issues & Improvements

### 1. LSP Errors Fixed (All Resolved)
- ✅ Fixed `TEST_MODE` initialization in `bot/config.py` (moved outside class)
- ✅ Added `Any` to typing imports
- ✅ Fixed return type for `get_railway_environment_info` to include bool
- ✅ Fixed type hint from 'any' to 'Any' in `get_deployment_context`
- ✅ Resolved duplicate `validate_hf_token` method

### 2. Missing Dependencies Added
- ✅ Added `asyncpg==0.30.0` to `requirements.txt` for Supabase PostgreSQL support
- ✅ All dependencies verified and tested

### 3. Deployment Configuration
- ✅ Procfile configured correctly: `web: python main.py`
- ✅ railway.toml with Nixpacks builder for Python 3.12.2
- ✅ Health check endpoint working at `/health` and `/health/json`
- ✅ Automatic PORT detection for Railway environment

---

## ⚠️ Pre-Deployment Requirements

### Critical: Update These Before Deploying

#### 1. HuggingFace API Token (REQUIRED for AI Features)
**Current Issue**: HF_TOKEN returns 401 Unauthorized

**Fix Required**:
```bash
# Get a new token from https://huggingface.co/settings/tokens
# Then set it in Railway:
railway variables set HF_TOKEN='hf_YourNewTokenHere'
```

#### 2. Supabase PostgreSQL URL (Optional - Hybrid Database)
**Current Issue**: DNS resolution fails in development

**Railway Configuration**:
```bash
# If using Supabase, ensure the URL is correct and accessible:
railway variables set SUPABASE_MGMT_URL='postgresql://user:pass@db.xxx.supabase.co:5432/postgres'
# OR use Railway's PostgreSQL database:
railway variables set DATABASE_URL='postgresql://...'
```

**Note**: MongoDB-only mode works fine. Supabase is optional for enhanced features.

---

## 🚀 Railway Deployment Steps

### Step 1: Prepare Environment Variables

**Required Variables** (Set in Railway Dashboard):
```bash
# Core Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
HF_TOKEN=hf_your_valid_token_here
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/database
ENCRYPTION_SEED=your_secure_32_character_key_here

# Admin Access
OWNER_ID=your_telegram_user_id
```

**Optional Variables**:
```bash
# Database (for hybrid mode)
SUPABASE_MGMT_URL=postgresql://...
# OR
DATABASE_URL=postgresql://...

# Performance Tuning
REQUEST_TIMEOUT=30
MAX_RETRIES=3
MAX_CHAT_HISTORY=20

# Environment
RAILWAY_ENVIRONMENT=production
```

### Step 2: Deploy to Railway

#### Option A: Using Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Link to your project
railway link

# Set environment variables
railway variables set TELEGRAM_BOT_TOKEN='your-token'
railway variables set HF_TOKEN='hf_your-token'
railway variables set MONGODB_URI='mongodb+srv://...'
railway variables set ENCRYPTION_SEED='your-secure-key'
railway variables set OWNER_ID='your-id'

# Deploy
railway up
```

#### Option B: Using Railway Dashboard
1. Go to [railway.app](https://railway.app)
2. Create a new project
3. Connect your GitHub repository
4. Go to Variables tab and add all required variables
5. Deploy will start automatically

### Step 3: Verify Deployment

**Check Health Status**:
```bash
# Simple health check
curl https://your-app.railway.app/health

# Detailed health check
curl https://your-app.railway.app/health/json
```

**Expected Response**:
- Status: `HEALTHY` (all systems operational)
- MongoDB: `healthy: true`
- HuggingFace API: `healthy: true` (after token update)
- Supabase: `healthy: true` or in fallback mode

---

## 🔍 Health Check Status Guide

### Healthy Status (HTTP 200)
```json
{
  "status": "healthy",
  "healthy": true,
  "degraded": false,
  "message": "All systems operational"
}
```
✅ Everything working perfectly

### Degraded Status (HTTP 200)
```json
{
  "status": "degraded",
  "healthy": true,
  "degraded": true,
  "message": "Core functionality operational, some features may be limited"
}
```
⚠️ Core bot works, but:
- Supabase might be unavailable (MongoDB fallback active)
- HuggingFace API issues (AI features limited)
- Railway will NOT restart the bot

### Unhealthy Status (HTTP 503)
```json
{
  "status": "unhealthy",
  "healthy": false,
  "message": "Critical functionality impaired"
}
```
❌ Critical issues (MongoDB failed, bot cannot function)
- Railway will restart the service

---

## 🛠️ Troubleshooting

### Issue 1: HuggingFace API Returns 401
**Cause**: Invalid or expired HF_TOKEN

**Fix**:
1. Visit https://huggingface.co/settings/tokens
2. Create new token (Read access)
3. Update Railway variable:
   ```bash
   railway variables set HF_TOKEN='hf_NewTokenHere'
   ```
4. Redeploy or restart service

### Issue 2: Supabase Connection Failed
**Cause**: DNS resolution or incorrect URL

**Fix Options**:
1. **Verify URL format**:
   ```
   postgresql://postgres:password@db.project.supabase.co:5432/postgres
   ```
2. **Use Railway PostgreSQL instead**:
   - Add PostgreSQL service in Railway
   - Use `DATABASE_URL` automatically provided
3. **Continue with MongoDB-only**:
   - Bot works perfectly with MongoDB alone
   - Supabase is optional for enhanced features

### Issue 3: Bot Not Responding to Commands
**Checks**:
1. Verify `TELEGRAM_BOT_TOKEN` is correct
2. Check Railway logs: `railway logs`
3. Ensure bot has proper permissions in Telegram
4. Verify health endpoint shows HEALTHY or DEGRADED (not UNHEALTHY)

### Issue 4: Port Binding Issues
**Solution**: Railway automatically sets `PORT` environment variable
- Health server automatically detects and uses Railway's PORT
- No manual configuration needed

---

## 📊 Production Deployment Checklist

- [ ] **Environment Variables Set**
  - [ ] TELEGRAM_BOT_TOKEN
  - [ ] HF_TOKEN (new, valid token)
  - [ ] MONGODB_URI (with SSL/TLS)
  - [ ] ENCRYPTION_SEED (secure 32+ char string)
  - [ ] OWNER_ID

- [ ] **Security Verified**
  - [ ] TEST_MODE is NOT set (or set to false)
  - [ ] ENCRYPTION_SEED is explicitly set (not auto-generated)
  - [ ] MongoDB uses SSL/TLS (mongodb+srv://)
  - [ ] All tokens are valid and active

- [ ] **Health Checks Pass**
  - [ ] `/health` returns 200 status
  - [ ] `/health/json` shows healthy or degraded status
  - [ ] MongoDB shows connected
  - [ ] HuggingFace API accessible (not 401)

- [ ] **Functionality Tests**
  - [ ] Bot responds to /start command
  - [ ] AI responses working (text generation)
  - [ ] File processing working (images, documents)
  - [ ] Admin commands accessible (if OWNER_ID set)

- [ ] **Railway Configuration**
  - [ ] Procfile present and correct
  - [ ] railway.toml configured
  - [ ] Health check path set to /health
  - [ ] Auto-deploy enabled (optional)

---

## 🔐 Security Best Practices

1. **Never commit secrets** to repository
2. **Use Railway's environment variables** for all sensitive data
3. **Rotate ENCRYPTION_SEED** carefully (requires data migration)
4. **Keep HF_TOKEN** updated and monitor usage
5. **Enable MongoDB authentication** and use SSL/TLS
6. **Set OWNER_ID** to restrict admin access
7. **Monitor logs** for security events

---

## 📈 Monitoring & Maintenance

### Railway Logs
```bash
# View real-time logs
railway logs

# Follow logs
railway logs --follow
```

### Health Monitoring
- Health check runs every 60 seconds
- Automatic restart on UNHEALTHY status
- Graceful degradation on partial failures

### Performance Metrics
- Response time target: <1 second
- Uptime target: 99%+
- Health check timeout: 30 seconds

---

## 📞 Support & Resources

- **Railway Docs**: https://docs.railway.app
- **Telegram Bot API**: https://core.telegram.org/bots/api
- **HuggingFace API**: https://huggingface.co/docs/api-inference
- **MongoDB Atlas**: https://docs.atlas.mongodb.com

---

## 🎉 Deployment Complete!

Once all environment variables are set and health checks pass:

1. Your bot will be live at the Railway-provided URL
2. Users can interact via Telegram
3. Health monitoring will ensure reliability
4. Auto-restart handles failures

**Test your deployment**:
```bash
# Send a message to your bot on Telegram
/start

# Or use the bot's username
@YourBotUsername
```

---

**Last Updated**: September 30, 2025
**Bot Version**: 2025.1.0
**Status**: ✅ Production Ready (after HF_TOKEN update)
