import google.generativeai as genai
import os

def test_gemini_2_flash():
    """Test Gemini 2.0 Flash model"""
    
    # Configure API
    GEMINI_API_KEY = "AIzaSyC5L0eAqKP6RgFlYez45Hvs0u5QqcT-2pw"
    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        # Try Gemini 2.0 Flash first
        print("Testing Gemini 2.0 Flash...")
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content("Hello! Can you confirm you are Gemini 2.0 Flash and working correctly?")
        print("✅ Gemini 2.0 Flash is working!")
        print(f"Response: {response.text[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini 2.0 Flash failed: {str(e)}")
        
        # Fallback to stable version
        try:
            print("Trying fallback: gemini-1.5-flash...")
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Test message")
            print("✅ Fallback to Gemini 1.5 Flash successful!")
            print("Update app.py to use 'gemini-1.5-flash' instead")
            return False
        except Exception as e2:
            print(f"❌ All models failed: {str(e2)}")
            return False

if __name__ == "__main__":
    test_gemini_2_flash()