#!/usr/bin/env python3
"""Test the RAG system fallback functionality"""

import json
import requests
import sys

# Test the API endpoint
BASE_URL = "http://localhost:5000"

def test_qa():
    """Test Q&A endpoint with a sample question"""
    
    # Use a known user and document from the logs
    user_id = "WSNAWALIspNg2HCKtXoDME5aQIH2"
    document_id = "7d760d79-a368-4d65-b060-498b010dfc48"  # AI.pdf
    question = "What is the history of AI?"
    
    payload = {
        "user_id": user_id,
        "document_id": document_id,
        "question": question
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/qa/ask", 
                                json=payload,
                                timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Response ===")
            print(f"Question: {result.get('question')}")
            print(f"\nAnswer: {result.get('answer')[:500]}...")
            print(f"\nQuestion Type: {result.get('question_type')}")
            
            rag_metadata = result.get('rag_metadata', {})
            if rag_metadata:
                print(f"\nRAG Metadata:")
                print(f"  - RAG Enabled: {rag_metadata.get('rag_enabled')}")
                print(f"  - Processing Method: {rag_metadata.get('processing_method')}")
                print(f"  - Has Error: {rag_metadata.get('has_error')}")
                if rag_metadata.get('has_error'):
                    print(f"  - Error: {rag_metadata.get('rag_error', '')[:100]}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_rag_stats():
    """Test RAG stats endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/rag/stats", timeout=5)
        print(f"\n=== RAG Stats ===")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Stats request failed: {e}")

if __name__ == "__main__":
    print("Testing RAG system with Gemini API quota issue...")
    print("=" * 60)
    
    test_rag_stats()
    print("\n" + "=" * 60)
    test_qa()