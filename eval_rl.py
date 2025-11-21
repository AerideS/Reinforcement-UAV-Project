# eval_rl.py
import asyncio

from px4_env import Px4Env
from agent import Agent

NUM_EVAL_EPISODES = 5  # 평가 몇 번 돌려볼지


async def eval_policy():
    env = Px4Env()
    agent = Agent()

    # 1) 이미 학습된 파라미터 로드
    agent.load("policy_final.pth")

    # 2) 탐험 거의 없게 (거의 greedy 정책 사용)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.05

    try:
        for ep in range(NUM_EVAL_EPISODES):
            print("\n==============================")
            print(f"       EVAL EPISODE {ep+1}/{NUM_EVAL_EPISODES}")
            print("==============================")

            obs = await env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            last_info = {}

            while not done:
                # 학습 X, 저장 X → 그냥 정책만 사용
                action = agent.select_action(obs)
                next_obs, reward, done, info = await env.step(action)

                obs = next_obs
                ep_reward += reward
                step += 1
                last_info = info

                if step > env.max_steps + 100:
                    print("[EVAL] Force break (step overflow)")
                    break

            reason = last_info.get("reason", "unknown")
            await env.close_episode()
            print(
                f"[EVAL] Episode {ep+1} finished: "
                f"steps={step}, return={ep_reward:.2f}, reason={reason}"
            )

    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(eval_policy())
