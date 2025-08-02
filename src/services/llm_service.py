import google.generativeai as genai
from openai import OpenAI, RateLimitError, AuthenticationError, APIConnectionError
import streamlit as st
import time

# Global variables to store configured API clients
_gemini_api_key = None
_openai_client = None
max_retries = 3

# Available models
openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-pro", "gemini-2.0-flash", "gemini-2.5-flash"]


def get_available_models(api_provider):
    """Get available models for the specified provider"""
    if api_provider == "OpenAI":
        return openai_models
    elif api_provider == "Gemini":
        return gemini_models
    return []


def configure_api_key(openai_key, gemini_key):
    """Configure API clients with proper error handling"""
    global _gemini_api_key, _openai_client

    if openai_key:
        try:
            _openai_client = OpenAI(api_key=openai_key)
            st.session_state.openai_available = True
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            _openai_client = None
            st.session_state.openai_available = False

    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            _gemini_api_key = gemini_key
            st.session_state.gemini_available = True
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {str(e)}")
            _gemini_api_key = None
            st.session_state.gemini_available = False


def generate_recipe_with_llm(api_provider, model_name, ingredients, meal_type, cuisine, dietary_restrictions,
                             cooking_time):
    """Generate recipe with robust error handling and retries"""
    if api_provider == "OpenAI" and not _openai_client:
        return "Error: OpenAI API key not configured"
    if api_provider == "Gemini" and not _gemini_api_key:
        return "Error: Gemini API key not configured"

    prompt = f"""Generate a detailed recipe with these specifications:
    - Ingredients: {ingredients}
    - Meal Type: {meal_type}
    - Cuisine: {cuisine}
    - Dietary Restrictions: {dietary_restrictions}
    - Max Cooking Time: {cooking_time} minutes

    MUST INCLUDE:
    1. A FUNNY short recipe name (include puns or pop culture references!)
    2. Short description (1 funny sentence)
    2.1. Include the calorie count( intend pun for the fitness freaks)
    3. "Chef's Confidential" section with 2 humorous tips
    4. Ingredients list with quantities (include funny measurements)
    5. Step-by-step instructions (add humorous commentary)
    6. Total time estimate
    7. Serving suggestion with a twist

    Format like a viral food blog post with emojis!
    """

    for attempt in range(max_retries):
        try:
            time.sleep(1)  # Basic rate limiting

            if api_provider == "OpenAI":
                response = _openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional chef assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content

            elif api_provider == "Gemini":
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.7}
                )
                return response.text

        except RateLimitError:
            wait = min(2 ** attempt, 10)  # Exponential backoff with max 10s
            st.warning(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        except AuthenticationError as e:
            return f"Error: Invalid API key for {api_provider}"

        except APIConnectionError as e:
            if attempt == max_retries - 1:
                return f"Error: Could not connect to {api_provider} API"
            time.sleep(2)
            continue

        except Exception as e:
            error_msg = str(e).lower()
            if "invalid" in error_msg or "not found" in error_msg:
                return f"Error: Invalid model '{model_name}' for {api_provider}"
            if attempt == max_retries - 1:
                return f"Error: Failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
            continue

    return f"Error: Failed to generate recipe after {max_retries} attempts"