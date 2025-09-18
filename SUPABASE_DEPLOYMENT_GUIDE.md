# Individual User Supabase Database System - Deployment Guide

## Overview

This system provides **complete user data isolation** by creating individual PostgreSQL schemas for each user within Supabase, ensuring that each user has their own isolated database environment with enterprise-grade security.

## Key Features

✅ **Individual User Databases**: Each user gets their own PostgreSQL schema (e.g., `user_123456789`)  
✅ **Automatic Provisioning**: New user schemas created automatically on first interaction  
✅ **Complete Data Isolation**: Users cannot access each other's data at any level  
✅ **Enterprise Security**: AESGCM-256 encryption with user-specific keys  
✅ **Row Level Security**: PostgreSQL RLS policies for additional protection  
✅ **Developer Separation**: Management database separate from user data  

## Architecture

```
┌─────────────────────┐    ┌─────────────────────────────┐
│   Management DB     │    │      User Schemas           │
│                     │    │                             │
│ ├── user_schemas    │    │ ├── user_123456789/         │
│ ├── system_metadata │    │ │   ├── api_keys            │
│ └── admin_logs      │    │ │   ├── conversations       │
│                     │    │ │   ├── files               │
└─────────────────────┘    │ │   └── preferences         │
                           │ │                            │
                           │ ├── user_987654321/         │
                           │ │   ├── api_keys            │
                           │ │   ├── conversations       │
                           │ │   └── ...                 │
                           │ └── user_555666777/         │
                           └─────────────────────────────┘
```

## Environment Variables

### Required Configuration

```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"

# Supabase Configuration  
SUPABASE_MGMT_URL="postgresql://postgres:[password]@[host]:5432/[database]"

# Optional: Separate user database URL (defaults to management URL)
SUPABASE_USER_BASE_URL="postgresql://postgres:[password]@[host]:5432/[user_database]"

# Optional: Custom encryption seed (auto-generated if not provided)
ENCRYPTION_SEED="your_secure_encryption_seed_here"
```

### Example Configuration

```bash
# Production Example
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
SUPABASE_MGMT_URL="postgresql://postgres:secure_password@db.supabase.co:5432/management_db"
ENCRYPTION_SEED="your-256-bit-encryption-seed-here"

# Development Example  
SUPABASE_MGMT_URL="postgresql://postgres:password@localhost:5432/dev_ai_bot"
```

## Supabase Setup

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Note your project URL and database password
3. Navigate to Settings → Database
4. Copy the "Connection string" under "Connection pooling"

### 2. Database Configuration

The system will automatically:
- Create management tables (`user_schemas`, `system_metadata`, `admin_logs`)
- Set up user schemas on first interaction
- Configure Row Level Security policies
- Initialize encryption system

### 3. Optional: Separate User Database

For maximum isolation, you can use a separate Supabase project for user data:

```bash
# Management database (stores user mappings only)
SUPABASE_MGMT_URL="postgresql://postgres:pass@mgmt.supabase.co:5432/mgmt"

# User database (stores actual user data)  
SUPABASE_USER_BASE_URL="postgresql://postgres:pass@users.supabase.co:5432/users"
```

## Migration from MongoDB

### Automatic Migration Detection

The system automatically detects your current configuration:

```python
# If you have both configured, Supabase takes precedence
MONGODB_URI="mongodb+srv://..."      # Legacy system
SUPABASE_MGMT_URL="postgresql://..." # New system (used)
```

### Manual Migration Steps

1. **Set up Supabase** following the steps above
2. **Add environment variables** while keeping MongoDB config
3. **Restart the bot** - it will auto-detect and use Supabase
4. **Verify functionality** with test users
5. **Remove MongoDB configuration** once satisfied

## Security Features

### User-Specific Encryption

Each user gets their own encryption keys derived from:
- Global seed (from environment or auto-generated)
- User's Telegram ID
- PBKDF2 with 50,000+ iterations

### PostgreSQL Schema Isolation

```sql
-- Example: User 123456789 gets schema "user_123456789"
CREATE SCHEMA user_123456789;
CREATE TABLE user_123456789.api_keys (...);
CREATE TABLE user_123456789.conversations (...);

-- Row Level Security
ALTER TABLE user_123456789.api_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_isolation_policy ON user_123456789.api_keys
  FOR ALL TO public
  USING (user_id = 123456789);
```

### Data Protection

- **API Keys**: AESGCM-256 encrypted with user-specific keys
- **Conversations**: Stored in isolated user schemas
- **Files**: User-specific tables with encryption
- **Preferences**: Schema-isolated with RLS policies

## Testing

Run the comprehensive test suite:

```bash
# Install dependencies
pip install asyncpg sqlalchemy[asyncio] psycopg2-binary

# Run isolation tests
python test_supabase_user_isolation.py
```

Expected output:
```
🎉 ALL TESTS PASSED - Individual user isolation system is working correctly!
✅ User schema isolation: PASSED
✅ Per-user encryption: PASSED  
✅ Data protection: PASSED
✅ Automatic provisioning: PASSED
✅ Factory integration: PASSED
✅ Configuration validation: PASSED
```

## Deployment

### Using Current Replit Environment

1. **Add Environment Variables** in Replit Secrets:
   ```
   SUPABASE_MGMT_URL = postgresql://postgres:[password]@[host]:5432/[database]
   ```

2. **The bot will automatically**:
   - Detect Supabase configuration
   - Switch from MongoDB to Supabase
   - Create user schemas on demand
   - Maintain all existing functionality

3. **Verify** with a test message to the bot

### Using External Deployment

1. Set environment variables in your deployment platform
2. Ensure network connectivity to Supabase
3. Deploy with the existing codebase - no additional changes needed

## Monitoring and Maintenance

### User Schema Management

Check created schemas:
```sql
SELECT user_id, schema_name, created_at, active 
FROM user_schemas 
ORDER BY created_at DESC;
```

### Encryption Status

The system logs encryption operations:
```
🔐 Using encryption seed from environment variables (secure)
🔒 Encryption system initialized with AESGCM-256
🏗️ Created new user schema: user_123456789 for user 123456789
🔒 Successfully saved encrypted API key for user 123456789 in isolated schema
```

### Performance Monitoring

- Each user gets their own database engine (connection pooling)
- Automatic schema creation is performed once per user
- Row Level Security adds minimal overhead
- User-specific encryption uses efficient AESGCM

## Troubleshooting

### Common Issues

**"Supabase dependencies not available"**
```bash
pip install asyncpg sqlalchemy[asyncio] psycopg2-binary
```

**"SUPABASE_MGMT_URL must be a valid PostgreSQL connection string"**
- Ensure URL starts with `postgresql://` or `postgres://`
- Check credentials and host accessibility

**"Encryption system initialization failed"**  
- Verify `ENCRYPTION_SEED` if provided
- Check database write permissions for management tables

### Debugging

Enable detailed logging:
```python
import logging
logging.getLogger('bot.storage.supabase_user_provider').setLevel(logging.DEBUG)
```

Check system status:
```python
from bot.storage.factory import get_provider_info
print(get_provider_info())
```

## Support

For issues specific to this individual user database system:
1. Check the test suite results
2. Verify Supabase connectivity
3. Review encryption seed configuration
4. Check user schema creation logs

The system maintains full backward compatibility with the existing bot functionality while providing enterprise-grade user data isolation.