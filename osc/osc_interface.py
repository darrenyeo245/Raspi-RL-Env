from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client
import threading
import numpy as np
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BROADCAST_IP = os.getenv("BROADCAST_IP", "255.255.255.255")
BROADCAST_PORT = int(os.getenv("BROADCAST_PORT", 9001))
RASPI_PORT = int(os.getenv("RASPI_PORT", 9001))

class OSCInterface:
    def __init__(self, enable_logging=True, log_path="logs/agent_osc.log"):
        self.actor_state = np.zeros(3, dtype=np.float32)
        self.media_command_state = np.zeros(3, dtype=np.float32)
        self.reward = 0.0

        self._state_pending = False
        self._reward_pending = False
        self._manual_reset_pending = False
        self._episode_end_pending = False
        self._training_stop_pending = False
        self._lock = threading.Condition()

        self.client = udp_client.SimpleUDPClient(BROADCAST_IP, BROADCAST_PORT, allow_broadcast=True)
        self.local_client = udp_client.SimpleUDPClient("127.0.0.1", RASPI_PORT)
        self.logger = self._setup_logger(enable_logging=enable_logging, log_path=log_path)

        dispatcher = Dispatcher()
        dispatcher.map("/adm/obj/101/xyz", self.state_handler)
        dispatcher.map("/adm/obj/1/xyz", self.media_command_handler)
        dispatcher.map("/reward", self.reward_handler)
        dispatcher.map("/episode/reset_manual", self.manual_reset_handler)
        dispatcher.map("/episode/end", self.episode_end_handler)
        dispatcher.map("/training/stop", self.training_stop_handler)
        dispatcher.map("/training/stop_save", self.training_stop_handler)


        self.server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", RASPI_PORT), dispatcher)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self._log_event("listener_started", "/", ["0.0.0.0, 9001"])

    @staticmethod
    def _setup_logger(enable_logging, log_path):
        logger = logging.getLogger("agent_osc")
        logger.propagate = False

        if not enable_logging:
            logger.handlers.clear()
            logger.setLevel(logging.CRITICAL)
            return logger

        if logger.handlers:
            return logger

        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(message)s")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def _log_event(self, direction, address, payload):
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("%s | %s | %s", direction, address, list(payload))

    def state_handler(self, address, *args):
        if len(args) >= 3:
            with self._lock:
                self.actor_state = np.array(args[:3], dtype=np.float32)
                self._state_pending = True
                self._lock.notify_all()
            self._log_event("recv", address, args[:3])

    def media_command_handler(self, address, *args):
        if len(args) >= 3:
            with self._lock:
                self.media_command_state = np.array(args[:3], dtype=np.float32)
                self._lock.notify_all()
            self._log_event("recv", address, args[:3])

    def reward_handler(self, address, *args):
        if not args:
            return
        with self._lock:
            self.reward += float(args[0])
            self._reward_pending = True
            self._lock.notify_all()
        self._log_event("recv", address, args[:1])

    def manual_reset_handler(self, address, *args):
        with self._lock:
            self._manual_reset_pending = True
            self._lock.notify_all()
        self._log_event("recv", address, args)

    def episode_end_handler(self, address, *args):
        with self._lock:
            self._episode_end_pending = True
            self._lock.notify_all()
        self._log_event("recv", address, args)

    def training_stop_handler(self, address, *args):
        with self._lock:
            self._training_stop_pending = True
            self._lock.notify_all()
        self._log_event("recv", address, args)


    def get_state(self, wait_for_new=False, timeout=None):
        return self.get_actor_state(wait_for_new=wait_for_new, timeout=timeout)

    def get_actor_state(self, wait_for_new=False, timeout=None):
        with self._lock:
            if wait_for_new and not self._state_pending:
                self._lock.wait(timeout=timeout)

            s = self.actor_state.copy()
            self._state_pending = False
            return s

    def get_media_command_state(self):
        with self._lock:
            return self.media_command_state.copy()

    def get_reward(self, wait_for_new=False, timeout=None):
        with self._lock:
            if wait_for_new and not self._reward_pending:
                self._lock.wait(timeout=timeout)

            r = float(self.reward)
            self.reward = 0.0
            self._reward_pending = False
            return r

    def send_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        payload = action.tolist()
        self.client.send_message("/adm/obj/1/xyz", payload)
        with self._lock:
            self.media_command_state = action.copy()
        self._log_event("send", "/adm/obj/1/xyz", payload)

    def send_reset(self, init_state):
        init_state = np.asarray(init_state, dtype=np.float32)
        payload = init_state.tolist()
        self.client.send_message("/episode/reset", payload)
        self._log_event("send", "/episode/reset", payload)

    def wait_for_feedback(self, timeout=None):
        with self._lock:
            while (
                    not self._reward_pending
                    and not self._manual_reset_pending
                    and not self._episode_end_pending
                    and not self._training_stop_pending
            ):
                self._lock.wait(timeout=timeout)

            reward = float(self.reward)
            manual_reset = self._manual_reset_pending
            episode_end = self._episode_end_pending
            training_stop = self._training_stop_pending

            self.reward = 0.0
            self._reward_pending = False
            self._manual_reset_pending = False
            self._episode_end_pending = False
            self._training_stop_pending = False

            return reward, manual_reset, episode_end, training_stop

    def send_training_status(self, active: bool, text: str = "default"):
        payload = [int(active), text]
        self.client.send_message("/training/status", payload)
        self._log_event("send", "/training/status", payload)