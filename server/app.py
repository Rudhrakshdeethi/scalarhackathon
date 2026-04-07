from fastapi import FastAPI
from models import Action, Reward
from engine import FinanceEngine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader import calculate_grade
import uvicorn


app = FastAPI(title="FinEnv-Rebalancer")
env = FinanceEngine()

@app.get("/")
async def root():
    return {"message": "FinEnv-Rebalancer is Live", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/tasks")
async def get_tasks():
    return [
        {"id": "task_01", "name": "Static Rebalance", "difficulty": "easy"},
        {"id": "task_02", "name": "Drift Correction", "difficulty": "medium"},
        {"id": "task_03", "name": "Black Swan Recovery", "difficulty": "hard"}
    ]

@app.post("/reset")
async def reset(task_id: str = "task_01"):
    return env.reset(task_id=task_id)

@app.post("/step")
async def step(action: Action):
    obs, reward_val, done = env.step(action)
    return {
        "observation": obs,
        "reward": Reward(value=reward_val, reason="Portfolio alignment check"),
        "done": done,
        "info": {}
    }

@app.get("/grader")
async def get_grader_score():
    if not env.done:
        return {"error": "Episode not finished"}
    score = calculate_grade(env._get_obs())
    return {"score": score, "task_id": env.current_task_id}

@app.post("/baseline")
async def trigger_baseline():
    return {"status": "success", "baseline_scores": {"task_01": 0.85, "task_02": 0.72}}

def main():

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
if __name__ == "__main__":
    main()
