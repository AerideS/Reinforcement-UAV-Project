# train_rl.py
import asyncio

from px4_env import Px4Env
from agent import Agent


NUM_EPISODES = 10  # 테스트용, 나중에 늘리면 됨


async def train():
    env = Px4Env()
    agent = Agent()

    try:
        for ep in range(NUM_EPISODES):
            print(f"\n==============================")
            print(f"       EPISODE {ep+1}/{NUM_EPISODES}")
            print(f"==============================")

            obs = await env.reset()
            done = False
            ep_reward = 0.0
            step = 0

            while not done:
                # 1) 행동 선택
                action = agent.select_action(obs)

                # 2) 환경에 적용
                next_obs, reward, done, info = await env.step(action)

                # 3) 경험 저장
                agent.store_transition(obs, action, reward, next_obs, done)

                # 4) 학습 스텝
                agent.train_step()

                obs = next_obs
                ep_reward += reward
                step += 1

                # 안전장치: 너무 오래 걸리면 강제 종료
                if step > env.max_steps + 100:
                    print("[TRAIN] Force break (step overflow)")
                    break

            await env.close_episode()
            print(f"[TRAIN] Episode {ep+1} finished: steps={step}, return={ep_reward}")

        # 학습 끝난 후 모델 저장
        agent.save("policy_final.pth")

    finally:
        # 마지막에 SITL까지 완전히 종료
        await env.close()


if __name__ == "__main__":
    asyncio.run(train())
