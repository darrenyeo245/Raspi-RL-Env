import gymnasium as gym
import numpy as np


class MediaEnv(gym.Env):
    def __init__(self, osc_interface, max_steps):
        super().__init__()
        self.osc = osc_interface

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.max_steps = int(max_steps)
        self._step_count = 0

    def _build_observation(self, actor_state):
        media_state = self.osc.get_media_command_state().astype(np.float32)
        return np.concatenate(
            (
                actor_state.astype(np.float32),
                media_state
            )
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.osc.send_action(action)

        reward, manual_reset, episode_end, training_stop = self.osc.wait_for_feedback(timeout=None)

        actor_state = self.osc.get_actor_state(wait_for_new=False).astype(np.float32)
        obs = self._build_observation(actor_state)

        self._step_count += 1
        terminated = bool(episode_end)
        truncated = bool(manual_reset or training_stop)
        max_steps_reached = False
        if self._step_count >= self.max_steps:
            truncated = True
            max_steps_reached = True

        if bool(training_stop):
            episode_end_reason = "training_stop"
        elif terminated:
            episode_end_reason = "episode_end"
        elif bool(manual_reset):
            episode_end_reason = "manual_reset"
        elif max_steps_reached:
            episode_end_reason = "max_steps"
        else:
            episode_end_reason = "none"

        info = {
            "manual_reset": bool(manual_reset),
            "episode_end": bool(episode_end),
            "max_steps_reached": bool(max_steps_reached),
            "episode_end_reason": episode_end_reason,
            "actor_state": actor_state.tolist(),
            "media_state": obs[3:6].tolist(),
            "training_stop": bool(training_stop),

        }
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        self.osc.send_reset(np.zeros(3, dtype=np.float32))

        # Primärquelle für den Zustand ist die Kamera (/adm/obj/101/xyz).
        actor_state = self.osc.get_actor_state(wait_for_new=True, timeout=1.0).astype(np.float32)

        obs = self._build_observation(actor_state)

        info = {
            "actor_state": actor_state.tolist(),
            "media_state": obs[3:6].tolist(),
        }
        return obs, info
