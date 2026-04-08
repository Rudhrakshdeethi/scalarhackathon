import os
import requests
import json
import time
from openai import OpenAI

# MANDATORY VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
# Use internal URL for validator, fallback to localhost
SERVER_URL = os.getenv("OPENENV_URL", "http://localhost:7860")

def main():
    if not API_KEY or not MODEL_NAME:
        print("Error: Missing API_KEY or MODEL_NAME environment variables.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # 1. MANDATORY START BLOCK
    TASK_ID = "task_01"
    print(f"[START] task={TASK_ID}", flush=True)

    try:
        # RESET the environment
        response = requests.post(f"{SERVER_URL}/reset", params={"task_id": TASK_ID}, timeout=15)
        response.raise_for_status()
        observation = response.json()
        
        done = False
        step = 1
        max_steps = 10
        total_reward = 0.0

        while not done and step <= max_steps:
            # 2. PROMPT BUILDING
            prompt = (
                f"You are a financial AI. Maintain a 50/50 Cash to Asset ratio. "
                f"State: {json.dumps(observation)}. "
                f"Reply with exactly one JSON action: "
                f"{{\"action_type\": \"buy/sell/hold\", \"asset_id\": \"BTC\", \"quantity\": float}}"
            )

            # 3. INFERENCE WITH RETRIES
            action = {"action_type": "hold", "asset_id": "BTC", "quantity": 0.0}
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        timeout=30.0
                    )
                    response_text = completion.choices[0].message.content or ""
                    # Cleanup JSON
                    clean_json = response_text.replace("```json", "").replace("```", "").strip()
                    action = json.loads(clean_json)
                    break
                except Exception as e:
                    print(f"DEBUG: LLM Attempt {attempt+1} failed: {e}", flush=True)
                    time.sleep(2)

            # 4. STEP EXECUTION
            step_resp = requests.post(f"{SERVER_URL}/step", json=action, timeout=15)
            step_resp.raise_for_status()
            step_result = step_resp.json()
            
            observation = step_result["observation"]
            reward = step_result["reward"]
            done = step_result.get("done", False)
            total_reward += reward

            # 5. MANDATORY STEP BLOCK
            print(f"[STEP] step={step} reward={reward}", flush=True)
            
            step += 1

        # 6. MANDATORY END BLOCK
        print(f"[END] task={TASK_ID} score={total_reward/max_steps} steps={step-1}", flush=True)

    except Exception as e:
        print(f"DEBUG: Critical error: {e}", flush=True)
        # Fail gracefully with END tag so validator can parse
        print(f"[END] task={TASK_ID} score=0.0 steps={step} error=true", flush=True)

if __name__ == "__main__":
    main()
