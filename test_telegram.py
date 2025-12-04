"""
Simple Telegram Credential Test Script
Run this FIRST to verify your credentials work!
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

print("=" * 70)
print("🔐 TELEGRAM CREDENTIAL TEST")
print("=" * 70)
print()

# Check if credentials exist
if not TELEGRAM_TOKEN:
    print("❌ ERROR: TELEGRAM_TOKEN not found in .env file")
    print("   Add this line to your .env file:")
    print("   TELEGRAM_TOKEN=your_bot_token_here")
    exit(1)

if not CHAT_ID:
    print("❌ ERROR: CHAT_ID not found in .env file")
    print("   Add this line to your .env file:")
    print("   CHAT_ID=your_chat_id_here")
    exit(1)

print(f"✅ TELEGRAM_TOKEN: {TELEGRAM_TOKEN[:10]}...{TELEGRAM_TOKEN[-5:]}")
print(f"✅ CHAT_ID: {CHAT_ID}")
print()

# Test 1: Verify bot token
print("🔍 Test 1: Verifying bot token...")
try:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('ok'):
            bot_info = data.get('result', {})
            print(f"✅ Bot token is VALID!")
            print(f"   Username: @{bot_info.get('username')}")
            print(f"   Name: {bot_info.get('first_name')}")
            print(f"   ID: {bot_info.get('id')}")
        else:
            print(f"❌ Bot token invalid: {data.get('description')}")
            exit(1)
    else:
        print(f"❌ HTTP Error {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Connection error: {e}")
    print("   Check your internet connection")
    exit(1)

print()

# Test 2: Verify chat access
print("🔍 Test 2: Verifying chat access...")
try:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getChat"
    params = {"chat_id": CHAT_ID}
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('ok'):
            chat_info = data.get('result', {})
            print(f"✅ Chat ID is VALID!")
            print(f"   Type: {chat_info.get('type')}")
            if chat_info.get('username'):
                print(f"   Username: @{chat_info.get('username')}")
            if chat_info.get('first_name'):
                print(f"   Name: {chat_info.get('first_name')} {chat_info.get('last_name', '')}")
        else:
            print(f"❌ Chat access failed: {data.get('description')}")
            print()
            print("   SOLUTION: You need to start a chat with your bot first!")
            print("   1. Open Telegram")
            print("   2. Search for your bot username")
            print("   3. Click 'START' or send any message")
            print("   4. Run this script again")
            exit(1)
    else:
        print(f"❌ HTTP Error {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Connection error: {e}")
    exit(1)

print()

# Test 3: Send test message
print("🔍 Test 3: Sending test message...")
try:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": "✅ SUCCESS! Your Telegram credentials are working perfectly!\n\nYou can now run your trading bot."
    }
    response = requests.post(url, json=data, timeout=10)
    
    if response.status_code == 200:
        result = response.json()
        if result.get('ok'):
            print(f"✅ Test message sent successfully!")
            print(f"   Message ID: {result.get('result', {}).get('message_id')}")
            print()
            print("🎉 CHECK YOUR TELEGRAM - you should see the test message!")
        else:
            print(f"❌ Failed to send: {result.get('description')}")
            exit(1)
    else:
        print(f"❌ HTTP Error {response.status_code}")
        print(f"   Response: {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Connection error: {e}")
    exit(1)

print()
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("Your Telegram credentials are configured correctly.")
print("You can now run your main trading bot.")
print()
