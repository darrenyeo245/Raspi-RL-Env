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
        self._reset_training_stats()

    def _reset_episode(self):
        self.ep_return = 0.0
        self.ep_cumulative_reward = 0.0
        self.ep_max_reward = float("-inf")
        self.ep_rewards = []
        self.ep_steps = 0

    def _reset_training_stats(self):
        self.train_episodes = 0
        self.train_total_steps = 0
        self.train_sum_returns = 0.0
        self.train_sum_cumulative_rewards = 0.0
        self.train_best_ep_reward = float("-inf")
        self.train_worst_ep_reward = float("inf")
        self.train_global_max_reward = float("-inf")
        self.train_global_min_reward = float("inf")
        self.train_reason_counts = {}
        self.train_policy_entropy_values = []
        self.train_policy_means = []
        self.train_policy_stds = []

    def _extract_policy_stats(self):
        obs = self.locals.get("new_obs")
        if obs is None:
            obs = self.locals.get("observations")

        if obs is None:
            return None, None, None

        obs_t = torch.as_tensor(obs)
        dist = self.model.policy.get_distribution(obs_t)

        policy_mean = dist.distribution.mean.detach().cpu().numpy()
        policy_std = dist.distribution.stddev.detach().cpu().numpy()
        entropy = float(dist.entropy().mean().item())
        return policy_mean, policy_std, entropy

    def _print_episode_summary(self, reason, gamma, policy_mean, policy_std, entropy):
        mean_reward = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0
        print(
            "Episode Summary\n"
            f"  reason           : {reason}\n"
            f"  length           : {self.ep_steps}\n"
            f"  gamma            : {gamma:.4f}\n"
            f"  return (disc.)   : {self.ep_return:.4f}\n"
            f"  cumulative reward: {self.ep_cumulative_reward:.4f}\n"
            f"  max_reward       : {self.ep_max_reward:.4f}\n"
            f"  mean_reward      : {mean_reward:.4f}\n"
            f"  policy_mean      : {None if policy_mean is None else policy_mean.tolist()}\n"
            f"  policy_std       : {None if policy_std is None else policy_std.tolist()}\n"
            f"  policy_entropy   : {entropy if entropy is not None else 'n/a'}\n"
            f"  total_steps      : {self.num_timesteps}\n"
        )

    def _print_training_summary(self):
        if self.train_episodes == 0:
            print(
                "Training Summary\n"
                "  episodes         : 0\n"
                f"  total_steps      : {self.num_timesteps}\n"
                "  Hinweis          : Keine auswertbaren Episoden abgeschlossen.\n"
            )
            return

        avg_return = self.train_sum_returns / self.train_episodes
        avg_cumulative = self.train_sum_cumulative_rewards / self.train_episodes
        avg_ep_len = self.train_total_steps / self.train_episodes
        avg_entropy = (
            float(np.mean(self.train_policy_entropy_values))
            if self.train_policy_entropy_values else float("nan")
        )

        policy_mean_avg = None
        policy_std_avg = None
        if self.train_policy_means:
            policy_mean_avg = np.mean(np.stack(self.train_policy_means, axis=0), axis=0).tolist()
        if self.train_policy_stds:
            policy_std_avg = np.mean(np.stack(self.train_policy_stds, axis=0), axis=0).tolist()

        print(
            "=== Training Summary ===\n"
            f"  episodes             : {self.train_episodes}\n"
            f"  total_steps          : {self.train_total_steps}\n"
            f"  avg_ep_length        : {avg_ep_len:.2f}\n"
            f"  avg_return (disc.)   : {avg_return:.4f}\n"
            f"  avg_cumulative_reward: {avg_cumulative:.4f}\n"
            f"  best_ep_reward       : {self.train_best_ep_reward:.4f}\n"
            f"  worst_ep_reward      : {self.train_worst_ep_reward:.4f}\n"
            f"  global_max_reward    : {self.train_global_max_reward:.4f}\n"
            f"  global_min_reward    : {self.train_global_min_reward:.4f}\n"
            f"  avg_policy_entropy   : {avg_entropy:.4f}\n"
            f"  policy_mean_avg      : {policy_mean_avg}\n"
            f"  policy_std_avg       : {policy_std_avg}\n"
            f"  end_reasons          : {self.train_reason_counts}\n"
            "========================"
        )

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]
        gamma = float(getattr(self.model, "gamma", 0.99))

        self.ep_return += (gamma ** self.ep_steps) * reward
        self.ep_cumulative_reward += reward
        self.ep_steps += 1
        self.ep_max_reward = max(self.ep_max_reward, reward)
        self.ep_rewards.append(reward)

        if not done:
            return True

        reason = info.get("episode_end_reason", "unknown")
        training_stop = bool(info.get("training_stop", False))

        policy_mean, policy_std, entropy = self._extract_policy_stats()

        self.train_episodes += 1
        self.train_total_steps += self.ep_steps
        self.train_sum_returns += self.ep_return
        self.train_sum_cumulative_rewards += self.ep_cumulative_reward
        self.train_best_ep_reward = max(self.train_best_ep_reward, self.ep_cumulative_reward)
        self.train_worst_ep_reward = min(self.train_worst_ep_reward, self.ep_cumulative_reward)
        self.train_global_max_reward = max(self.train_global_max_reward, self.ep_max_reward)
        self.train_global_min_reward = min(self.train_global_min_reward, min(self.ep_rewards) if self.ep_rewards else reward)
        self.train_reason_counts[reason] = self.train_reason_counts.get(reason, 0) + 1

        if entropy is not None:
            self.train_policy_entropy_values.append(entropy)
        if policy_mean is not None:
            self.train_policy_means.append(policy_mean)
        if policy_std is not None:
            self.train_policy_stds.append(policy_std)

        if (not training_stop) and reason in ("manual_reset", "max_steps", "episode_end"):
            self.model.save(self.final_model_path)
            print("Episode gespeichert. Neue Episode wird gestartet")
            self._print_episode_summary(reason, gamma, policy_mean, policy_std, entropy)

        if training_stop:
            training_stop_save = bool(info.get("training_stop_save", False))
            if training_stop_save:
                self.model.save(self.final_model_path)
                print ("Training stop: Modell gespeichert, Training wird beendet.", flush=True)
            else:
                print("Training stop: Modell nicht gespeichert.", flush=True)
            self._print_training_summary()
            self._reset_episode()
            return False

        self._reset_episode()
        return True

def train(out_dir=None, total_timesteps=10000, algo="ppo", max_steps=100):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(out_dir, exist_ok=True)

    osc = OSCInterface()
    env = MediaEnv(osc, max_steps=max_steps)

    final_model_path = os.path.join(out_dir, "final_model")

    if os.path.exists(final_model_path + ".zip"):
        model = PPO.load(final_model_path, env=env, n_steps=64, batch_size=32)
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
        print("Training beendet, Modell gespeichert als 'final_model.zip'.", flush=True)

    return model, env


if __name__ == "__main__":
    train()
