#!/usr/bin/env python3
"""
Script to test the Qwen API connection
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.qwen_client import QwenClient
from backend.app.config import Config

async def main():
    # Initialize the Qwen client
    client = QwenClient()
    
    # Test message
    test_message = "Hello, this is a test message to check if the Qwen API is working properly."
    
    print("Testing Qwen API connection...")
    
    try:
        # Test the connection
        messages = [
            {"role": "user", "content": test_message}
        ]
        
        response = await client.generate_response(messages)
        print(f"Qwen API test successful!")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error testing Qwen API: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())