import time
import os
from datetime import datetime

from pathlib import Path
import streamlit as st

from services import (configure_api_key, generate_recipe_with_llm, get_available_models)


def init_session():
    """Initialize Streamlit session state variables"""
    if 'generating_recipe' not in st.session_state:
        st.session_state.generating_recipe = False
    if 'last_request_time' not in st.session_state:
        st.session_state.last_request_time = 0
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {
            'OpenAI': False,
            'Gemini': False
        }
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None


def check_rate_limit():
    """Enforce rate limiting between API calls"""
    current_time = time.time()
    elapsed = current_time - st.session_state.last_request_time
    if elapsed < 1.5:  # 1.5 second cooldown between requests
        time.sleep(1.5 - elapsed)
    st.session_state.last_request_time = time.time()


def reset_state():
    """Reset generation state and clear errors"""
    st.session_state.generating_recipe = False
    st.session_state.error_message = None


def save_recipe(recipe_output, recipe_name):
    """Save recipe in multiple formats under output/ folder"""
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Sanitize recipe name for filename
    sanitized_name = "".join(c if c.isalnum() else "_" for c in recipe_name if c.isalnum() or c in "_-")
    timestamp = datetime.now().strftime("%Y%m%d")
    base_filename = f"{output_dir}/{sanitized_name}_{timestamp}"

    # Save as HTML (best for emojis)
    html_path = f"{base_filename}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{recipe_name}</title>
            <style>
                body {{ font-family: Arial; padding: 20px; }}
                .recipe {{ 
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    max-width: 800px;
                    margin: 0 auto;
                }}
                h1 {{ color: #e67e22; }}
                pre {{ white-space: pre-wrap; font-family: inherit; }}
            </style>
        </head>
        <body>
            <div class="recipe">
                <h1>🍳 {recipe_name}</h1>
                <pre>{recipe_output}</pre>
            </div>
        </body>
        </html>
        """)

    # Save as plain text
    txt_path = f"{base_filename}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(recipe_output)

    return html_path, txt_path


def display_recipe(recipe_output):
    """Beautifully formatted recipe display with save options"""
    if not recipe_output or recipe_output.startswith("Error:"):
        st.session_state.error_message = recipe_output or "Failed to generate recipe"
        return

    # Clear any previous errors
    st.session_state.error_message = None

    # Split the recipe output into lines
    recipe_lines = recipe_output.split('\n')
    if not recipe_lines:
        return

    # Extract recipe name (first line)
    recipe_name = recipe_lines[0].strip('"')

    # Display recipe with custom styling
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                <h2 style="color: #2d3436; margin-bottom: 0.5rem; text-align: center;">
                    🍳 Your Custom Recipe
                </h2>
                <h3 style="color: #e17055; margin-top: 0; text-align: center; font-style: italic;">
                    "{recipe_name}"
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display the recipe content
        st.markdown('\n'.join(recipe_lines[1:]))

        # Save buttons
        st.markdown("---")
        st.subheader("📤 Save Recipe")
        try:
            html_path, txt_path = save_recipe(recipe_output, recipe_name)

            col1, col2, col3 = st.columns(3)
            with col1:
                with open(html_path, "rb") as f:
                    st.download_button(
                        "💾 Save as HTML",
                        data=f,
                        file_name=f"{recipe_name}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            with col2:
                with open(txt_path, "rb") as f:
                    st.download_button(
                        "📝 Save as Text",
                        data=f,
                        file_name=f"{recipe_name}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Failed to save recipe: {str(e)}")


def main():
    # Initialize session state
    init_session()

    # Load API keys
    openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    # Configure API clients
    configure_api_key(openai_key, gemini_key)

    # Verify API availability
    if not openai_key and not gemini_key:
        st.error("No API keys configured! Please set environment variables.")
        return

    # --- UI Layout ---
    st.set_page_config(layout="centered", page_title="AI Recipe Generator")
    st.title("👨‍🍳 AI Recipe Generator")
    st.markdown("Enter your ingredients and preferences for a custom recipe!")

    # Clear previous errors when starting new generation
    if st.session_state.generating_recipe:
        st.session_state.error_message = None

    # Provider selection with availability check
    available_providers = []
    if openai_key:
        available_providers.append("OpenAI")
    if gemini_key:
        available_providers.append("Gemini")

    if not available_providers:
        st.error("No available API providers. Please configure at least one API key.")
        return

    api_provider_choice = st.radio(
        "Choose AI Provider:",
        available_providers,
        index=0,
        horizontal=True
    )

    # Model selection
    available_models = get_available_models(api_provider_choice)
    if not available_models:
        st.error(f"No models available for {api_provider_choice}")
        return

    selected_model = st.selectbox("Choose AI Model:", available_models)

    # Recipe form
    with st.form("recipe_form"):
        ingredients = st.text_area(
            "🍅 Ingredients You Have",
            placeholder="chicken, rice, vegetables...",
            height=100,
            key="ingredients_input",
            help="What's in your fridge/pantry?"
        )

        col1, col2 = st.columns(2)
        with col1:
            meal_type = st.selectbox(
                "⏰ Meal Type",
                ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert"],
                key="meal_type_select",
                help="When are you eating this?"
            )
            cuisine = st.text_input(
                "🌍 Cuisine Preferences",
                "Any",
                key="cuisine_input",
                help="e.g., Italian, Mexican, Indian"
            )

        with col2:
            dietary_restrictions = st.text_input(
                "🚫 Dietary Restrictions",
                "None",
                key="dietary_input",
                help="e.g., vegetarian, gluten-free"
            )
            cooking_time = st.slider(
                "⏱️ Max Cooking Time (minutes)",
                5, 180, 30, step=5,
                key="cooking_time_slider"
            )

        submitted = st.form_submit_button(
            "✨ Generate Recipe ✨",
            disabled=st.session_state.generating_recipe,
            type="primary"
        )

        if submitted:
            if not ingredients.strip():
                st.warning("Please enter at least one ingredient!")
            else:
                st.session_state.generating_recipe = True
                st.session_state.error_message = None
                check_rate_limit()

    # Display any errors
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    # Handle recipe generation
    if st.session_state.generating_recipe and ingredients.strip():
        with st.spinner("🧑‍🍳 Cooking up your recipe..."):
            try:
                recipe_output = generate_recipe_with_llm(
                    api_provider=api_provider_choice,
                    model_name=selected_model,
                    ingredients=ingredients,
                    meal_type=meal_type,
                    cuisine=cuisine,
                    dietary_restrictions=dietary_restrictions,
                    cooking_time=cooking_time
                )
                display_recipe(recipe_output)

            except Exception as e:
                st.session_state.error_message = f"Recipe generation failed: {str(e)}"
            finally:
                reset_state()

    # Footer
    st.markdown("---")
    st.caption("🍳 Powered by AI Chef Assistants")
    st.caption("👨‍💻 Created by Sai Prathap Reddy Cheluri")


if __name__ == "__main__":
    main()