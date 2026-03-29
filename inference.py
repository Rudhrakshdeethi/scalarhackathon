import os
import requests
import json
from openai import OpenAI

# 1. Configuration
# The validator uses your Space's URL or localhost:7860
BASE_URL = os.getenv("OPENENV_URL", "http://localhost:7860")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def run_inference(task_id="task_01"):
    # RESET: Start the environment
    response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    obs = response.json()
    done = False
    
    print(f"Started {task_id}")

    # LOOP: Step until finished
    while not done:
        # Construct the prompt for Gemini
        prompt = f"Current State: {obs}. Maintain 50/50 cash-to-asset ratio. Output JSON: {{'action_type': 'buy/sell/hold', 'asset_id': 'BTC', 'quantity': float}}"
        
        completion = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        action = json.loads(completion.choices[0].message.content)
        
        # STEP: Send action to your FastAPI server
        step_resp = requests.post(f"{BASE_URL}/step", json=action).json()
        obs = step_resp["observation"]
        done = step_resp["done"]
        
    # FINAL: Get score from your grader endpoint
    final_score = requests.get(f"{BASE_URL}/grader").json()
    print(f"Final Score: {final_score}")

if __name__ == "__main__":
    run_inference()