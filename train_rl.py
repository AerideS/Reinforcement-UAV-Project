# train_rl.py
import asyncio
import csv

from px4_env import Px4Env
from agent import Agent


NUM_TRAIN_EPISODES = 100   # 학습 에피소드 수 (나중에 늘리면 됨)
NUM_EVAL_EPISODES = 3     # 학습 후 평가 에피소드 수


async def train():
    env = Px4Env()
    agent = Agent()

    # 에피소드별 통계 기록용
    episode_stats = []

    try:
        # ============================
        # 1) 학습 루프
        # ============================
        for ep in range(NUM_TRAIN_EPISODES):
            print("\n==============================")
            print(f"       TRAIN EPISODE {ep+1}/{NUM_TRAIN_EPISODES}")
            print("==============================")

            obs = await env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            last_info = {}

            while not done:
                # 1) 행동 선택
                action = agent.select_action(obs)

                # 2) 환경에 적용
                next_obs, reward, done, info = await env.step(action)
                last_info = info  # 종료 이유 추적용

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

            reason = last_info.get("reason", "unknown")

            await env.close_episode()
            print(
                f"[TRAIN] Episode {ep+1} finished: "
                f"steps={step}, return={ep_reward:.2f}, reason={reason}"
            )

            # 통계 저장
            episode_stats.append(
                {
                    "ep": ep + 1,
                    "steps": step,
                    "return": ep_reward,
                    "reason": reason,
                }
            )

        # ============================
        # 2) 모델 저장
        # ============================
        agent.save("policy_final.pth")

        # ============================
        # 3) 학습 결과 요약 출력
        # ============================
        if episode_stats:
            avg_return = sum(s["return"] for s in episode_stats) / len(episode_stats)
            avg_steps = sum(s["steps"] for s in episode_stats) / len(episode_stats)
            num_goal = sum(1 for s in episode_stats if s["reason"] == "goal")
            num_collision = sum(1 for s in episode_stats if s["reason"] == "collision")
            num_timeout = sum(1 for s in episode_stats if s["reason"] == "timeout")

            print("\n===== TRAIN SUMMARY =====")
            print(f"Episodes        : {len(episode_stats)}")
            print(f"Avg Return      : {avg_return:.2f}")
            print(f"Avg Steps       : {avg_steps:.2f}")
            print(f"Goal Reached    : {num_goal}")
            print(f"Collision       : {num_collision}")
            print(f"Timeout         : {num_timeout}")
            print("=========================\n")

            # CSV로 저장해서 나중에 그래프 그려볼 수 있게
            with open("train_log.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "steps", "return", "reason"])
                for s in episode_stats:
                    writer.writerow([s["ep"], s["steps"], s["return"], s["reason"]])
            print("[TRAIN] Saved train_log.csv")

        # ============================
        # 4) 평가 에피소드 (no training)
        # ============================
        print(f"[EVAL] Running {NUM_EVAL_EPISODES} evaluation episodes (no training)...")

        # 탐험 줄이기 (에이전트에 set_eval_mode가 있으면 사용)
        if hasattr(agent, "set_eval_mode"):
            agent.set_eval_mode(epsilon=0.05)
        elif hasattr(agent, "epsilon"):
            agent.epsilon = 0.05  # 최소한 탐험 줄이기

        for ep in range(NUM_EVAL_EPISODES):
            print("\n------------------------------")
            print(f"       EVAL EPISODE {ep+1}/{NUM_EVAL_EPISODES}")
            print("------------------------------")

            obs = await env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            last_info = {}

            while not done:
                # 평가에서는 train_step, store_transition 호출 X
                action = agent.select_action(obs)
                next_obs, reward, done, info = await env.step(action)
                last_info = info

                obs = next_obs
                ep_reward += reward
                step += 1

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
        # 마지막에 SITL까지 완전히 종료
        await env.close()


if __name__ == "__main__":
    asyncio.run(train())
