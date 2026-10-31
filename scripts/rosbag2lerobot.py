#!/usr/bin/env python3

import importlib.util
import shutil
import yaml
import numpy as np
import torch

from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


ODOM_TOPIC = "/model/demoRobot/odometry"
CMD_TOPIC = "/cmd_vel"


def load_config_py(config_path):
    spec = importlib.util.spec_from_file_location(
        "diff_drive_features", config_path
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class RosbagToLeRobot:

    def __init__(self, bag_dir, output_dir, config_path, target_freq=20):
        self.bag_path = Path(bag_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_freq = target_freq

        self.episode_dirs = sorted(
            {p.parent for p in self.bag_path.rglob("*.db3")},
            key=lambda p: (
                int(p.name[2:]) if p.name.startswith("ep") and p.name[2:].isdigit()
                else p.name
            ),
        )

        if not self.episode_dirs:
            raise FileNotFoundError(f"No rosbag2 episodes found: {bag_dir}")

        config = load_config_py(config_path)

        self.features = config.features
        self.robot_type = config.robot_type
        self.repo_id = config.repo_id
        self.task_name = config.task_name

        print(f"Found {len(self.episode_dirs)} episodes")

        if self.output_dir.exists():
            ans = input(
                f"{self.output_dir} already exists. Overwrite? (y/n): "
            ).strip().lower()

            if ans != "y":
                raise RuntimeError("Conversion cancelled.")

            shutil.rmtree(self.output_dir)

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.target_freq,
            root=self.output_dir,
            robot_type=self.robot_type,
            features=self.features,
            use_videos=False,
        )

        self.dataset.meta.add_task(self.task_name)

    def get_messages(self, episode_dir):
        messages = {
            ODOM_TOPIC: [],
            CMD_TOPIC: [],
        }

        with Reader(episode_dir) as reader:
            connections = [
                c for c in reader.connections
                if c.topic in messages
            ]

            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                messages[connection.topic].append((msg, timestamp))

        for topic in messages:
            messages[topic].sort(key=lambda x: x[1])

        return messages

    @staticmethod
    def quaternion_to_yaw(q):
        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(sin_yaw, cos_yaw)

    @staticmethod
    def sample_and_hold(times, values, target_times):
        times = np.asarray(times)
        values = np.asarray(values)

        indices = np.searchsorted(
            times, target_times, side="right"
        ) - 1

        indices = np.clip(indices, 0, len(times) - 1)
        return values[indices]

    def convert_episode(self, episode_dir, episode_index):
        messages = self.get_messages(episode_dir)

        odom_messages = messages[ODOM_TOPIC]
        cmd_messages = messages[CMD_TOPIC]

        if not odom_messages:
            raise ValueError(f"No odometry messages in {episode_dir}")

        if not cmd_messages:
            raise ValueError(f"No cmd_vel messages in {episode_dir}")

        odom_times = np.array([ts for _, ts in odom_messages])
        cmd_times = np.array([ts for _, ts in cmd_messages])

        # Odometry state:
        # [x, y, sin(yaw), cos(yaw), linear_velocity, angular_velocity]
        states = []

        for msg, _ in odom_messages:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            twist = msg.twist.twist

            yaw = self.quaternion_to_yaw(orientation)

            states.append([
                position.x,
                position.y,
                np.sin(yaw),
                np.cos(yaw),
                twist.linear.x,
                twist.angular.z,
            ])

        states = np.asarray(states, dtype=np.float32)

        # Action:
        # [linear_velocity, angular_velocity]
        actions = []

        for msg, _ in cmd_messages:
            actions.append([
                msg.linear.x,
                msg.angular.z,
            ])

        actions = np.asarray(actions, dtype=np.float32)

        # Use the interval where both odometry and commands are available.
        start_time = max(odom_times[0], cmd_times[0])
        end_time = odom_times[-1]

        if end_time <= start_time:
            raise ValueError(f"Invalid time interval in {episode_dir}")

        interval_ns = int(1e9 / self.target_freq)

        target_times = np.arange(
            start_time,
            end_time + 1,
            interval_ns,
            dtype=np.int64,
        )

        sampled_states = self.sample_and_hold(
            odom_times, states, target_times
        )

        sampled_actions = self.sample_and_hold(
            cmd_times, actions, target_times
        )

        if len(target_times) == 0:
            raise ValueError(f"No frames generated in {episode_dir}")

        for state, action in zip(sampled_states, sampled_actions):
            self.dataset.add_frame({
                "observation.state": torch.tensor(
                    state, dtype=torch.float32
                ),
                "observation.environment_state": torch.tensor(
                    state, dtype=torch.float32
                ),
                "action": torch.tensor(
                    action, dtype=torch.float32
                ),
                "task": self.task_name,
            })

        self.dataset.save_episode()

        print(
            f"Episode {episode_index}: "
            f"{len(target_times)} frames saved"
        )

    def convert(self):
        for index, episode_dir in enumerate(self.episode_dirs):
            print(f"\nProcessing {episode_dir}")

            try:
                self.convert_episode(episode_dir, index)
            except Exception as e:
                print(f"Error processing {episode_dir}: {e}")
                raise

        return str(self.output_dir)


def main():
    with open("configs/conversion_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    converter = RosbagToLeRobot(
        bag_dir=config["bag_dir"],
        output_dir=config["output_dir"],
        config_path=config["config"],
        target_freq=config.get("target_freq", 20),
    )

    output_path = converter.convert()
    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
