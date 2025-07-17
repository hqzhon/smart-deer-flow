#!/usr/bin/env python3
import requests
import json

def test_chat_stream():
    url = "http://localhost:8000/api/chat/stream"
    
    payload = {
        "messages": [{"role": "user", "content": "Search for the latest news about artificial intelligence developments in 2024"}],
        "thread_id": "test-thread",
        "resources": [],
        "max_plan_iterations": 1,
        "max_step_num": 3,
        "max_search_results": 3,
        "auto_accepted_plan": True,
        "interrupt_feedback": "",
        "mcp_settings": {},
        "enable_background_investigation": False,
        "report_style": "academic",
        "enable_deep_thinking": False,
        "enable_collaboration": False,
        "enable_parallel_execution": False,
        "max_parallel_tasks": 1
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Sending request to /api/chat/stream...")
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            return
        
        print("\nStreaming response:")
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"Line: {line}")
                
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_chat_stream()