import os
import requests
from openai import OpenAI

# 1. Setup Gemini Client (OpenAI-compatible)
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 2. Local or HF Space URL (Change to your Space URL when deployed)
BASE_URL = "http://127.0.0.1:7860"

def run_task(task_id):
    print(f"--- Starting {task_id} ---")
    
    # Reset Environment
    obs = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}).json()
    done = False
    
    while not done:
        # Ask Gemini for the next move
        prompt = f"""
        You are a financial AI. 
        Current Prices: {obs['prices']}
        Current Holdings: {obs['holdings']}
        Current Cash: {obs['cash']}
        
        Goal: Maintain a 50/50 Cash-to-Asset ratio.
        Output ONLY a JSON action like this: 
        {{"action_type": "buy", "asset_id": "BTC", "quantity": 0.1}}
        """
        
        response = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        import json
        action = json.loads(response.choices[0].message.content)
        
        # Send action to Environment
        step_res = requests.post(f"{BASE_URL}/step", json=action).json()
        obs = step_res['observation']
        done = step_res['done']
        
        print(f"Step: {obs['step_count']} | Value: {obs['portfolio_value']} | Reward: {step_res['reward']['value']}")

    # Get Final Grade
    grade = requests.get(f"{BASE_URL}/grader").json()
    print(f"Final Score for {task_id}: {grade['score']}")

if __name__ == "__main__":
    # Test on the Easy Task
    run_task("task_01")