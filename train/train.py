import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from env.media_env import MediaEnv
from osc.osc_interface import OSCInterface
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch

class EpisodeSummaryCallback(BaseCallback):
    def __init__(self, final_model_path, verbose=0):
        super().__init__(verbose)
        self.final_model_path = final_model_path
        self._reset_episode()

    def _reset_episode(self):
        self.ep_return = 0.0
        self.ep_cumulative_reward = 0.0
        self.ep_max_reward = float("-inf")
        self.ep_rewards = []
        self.ep_steps = 0

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        gamma = self.model.gamma

        self.ep_return += (gamma ** self.ep_steps) * reward
        self.ep_cumulative_reward += reward
        self.ep_steps += 1
        self.ep_len = self.ep_steps
        self.ep_max_reward = max(self.ep_max_reward, reward)
        self.ep_rewards.append(reward)

        if done:
            reason = info.get("episode_end_reason", "unknown")
            training_stop = bool(info.get("training_stop", False))

            # Nur loggen, wenn manual_reset oder max_steps erreicht wurden
            if (not training_stop) and reason in ("manual_reset", "max_steps", "episode_end"):
                self.model.save(self.final_model_path)
                print("Episode gespeichert. Neue Episode wird gestartet")

                # Policy‑Stats aus dem letzten Beobachtungszustand
                # In SB3 liegt die neue Beobachtung meistens in `new_obs`
                obs = self.locals.get("new_obs")
                if obs is None:
                    obs = self.locals.get("observations")

                policy_mean = None
                policy_std = None
                entropy = None
                if obs is not None:
                    obs_t = torch.as_tensor(obs)
                    dist = self.model.policy.get_distribution(obs_t)
                    policy_mean = dist.distribution.mean.detach().cpu().numpy().tolist()
                    policy_std = dist.distribution.stddev.detach().cpu().numpy().tolist()
                    entropy = dist.entropy().mean().item()
                mean_reward = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0

                print(
                    "Episode Summary\n"
                    f"  reason           : {reason}\n"
                    f"  length           : {self.ep_len}\n"
                    f"  gamma            : {gamma:.4f}\n"  # NEU
                    f"  return (disc.)   : {self.ep_return:.4f}\n"  # diskontiert
                    f"  cumulative reward: {self.ep_cumulative_reward:.4f}\n"  # NEU: roh
                    f"  max_reward       : {self.ep_max_reward:.4f}\n"
                    f"  mean_reward      : {mean_reward:.4f}\n"
                    f"  policy_mean      : {policy_mean}\n"
                    f"  policy_std       : {policy_std}\n"
                    f"  policy_entropy   : {entropy:.4f}\n"
                    f"  total_steps      : {self.num_timesteps}\n"
                )

            self._reset_episode()

            if training_stop:
                self.model.save(self.final_model_path)
                print("Training stop: Modell gespeichert, Training wird beendet.", flush=True)
                return False

        return True

def train(out_dir='models', total_timesteps=10000, algo="ppo", max_steps=100):
    os.makedirs(out_dir, exist_ok=True)

    osc = OSCInterface()
    env = MediaEnv(osc, max_steps=max_steps)

    final_model_path = os.path.join(out_dir, "final_model")

    if os.path.exists(final_model_path + ".zip"):
        model = PPO.load(final_model_path, env=env)
        model.n_steps=64
        model.batch_size=32
        model._setup_model()
        print("Modell 'final_model' wurde geladen")
    else:
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            n_steps=64,
            batch_size=32,
        )

    print("Training gestartet. Jeder Schritt wartet auf manuelles Feedback (/reward, /episode/*).")
    print(f"Algorithmus: {algo.upper()} | Timesteps: {total_timesteps}")
    callback = EpisodeSummaryCallback(final_model_path=final_model_path)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        model.save(final_model_path)
        print("Training beendet, Modell gespeichert als 'final_model.zip'.", flush=True)

    return model, env


if __name__ == "__main__":
    train()
