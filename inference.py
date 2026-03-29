import os
import requests
import json
from openai import OpenAI

# MANDATORY VARIABLES FROM THE EXAMPLE
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# YOUR SERVER URL (Hugging Face internal or localhost)
# The validator usually sets OPENENV_URL, otherwise use your Space's direct URL
SERVER_URL = os.getenv("OPENENV_URL", "http://localhost:7860")

def main():
    # MANDATORY: Use OpenAI Client as per instructions
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # 1. RESET: Start the financial task
    # We'll default to task_01 unless specified
    try:
        response = requests.post(f"{SERVER_URL}/reset", params={"task_id": "task_01"})
        observation = response.json()
        print(f"Episode started: {observation}")
    except Exception as e:
        print(f"Failed to connect to server at {SERVER_URL}: {e}")
        return

    done = False
    step = 1
    max_steps = 10 # Adjust as needed

    while not done and step <= max_steps:
        # 2. PROMPT: Format the financial data for Gemini
        prompt = f"""
        Step: {step}
        Goal: Maintain 50/50 Cash to Asset Ratio.
        State: {observation}
        Reply with exactly one JSON action: {{"action_type": "buy/sell/hold", "asset_id": "BTC", "quantity": float}}
        """

        # 3. INFERENCE: Call the LLM using the mandatory client
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        response_text = completion.choices[0].message.content
        
        # Simple cleanup if model adds markdown backticks
        action_json = response_text.replace("```json", "").replace("```", "").strip()
        action = json.loads(action_json)

        print(f"Step {step}: Model suggested -> {action}")

        # 4. STEP: Post the action to your FastAPI environment
        step_result = requests.post(f"{SERVER_URL}/step", json=action).json()
        
        observation = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]

        print(f"Reward: {reward} | Done: {done}")
        step += 1

if __name__ == "__main__":
    main()
