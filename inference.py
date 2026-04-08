import os
import requests
import json
import time
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
SERVER_URL = os.getenv("OPENENV_URL", "http://localhost:7860")

def main():
    if not API_KEY or not MODEL_NAME:
        print("Error: Missing API_KEY or MODEL_NAME environment variables.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    try:
        response = requests.post(f"{SERVER_URL}/reset", params={"task_id": "task_01"}, timeout=10)
        response.raise_for_status()
        observation = response.json()
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return

    done = False
    step = 1
    max_steps = 10 

    while not done and step <= max_steps:
        prompt = f"Goal: Maintain 50/50 Cash to Asset Ratio. State: {observation}. Reply with exactly one JSON action: {{\"action_type\": \"buy/sell/hold\", \"asset_id\": \"BTC\", \"quantity\": float}}"

        action = {"action_type": "hold", "asset_id": "BTC", "quantity": 0.0} # Default Fallback
        
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=30.0 
                )
                response_text = completion.choices[0].message.content
                clean_json = response_text.replace("```json", "").replace("```", "").strip()
                action = json.loads(clean_json)
                break
            except Exception as e:
                print(f"LLM Attempt {attempt+1} failed: {e}")
                time.sleep(2)
        try:
            print(f"Step {step}: Executing -> {action}")
            step_resp = requests.post(f"{SERVER_URL}/step", json=action, timeout=10)
            step_resp.raise_for_status()
            step_result = step_resp.json()
            
            observation = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]
            print(f"Reward: {reward} | Done: {done}")
        except Exception as e:
            print(f"Step failed: {e}")
            break 
            
        step += 1

if __name__ == "__main__":
    main()