# Physical AI Training

ROS 2、rosbag2、LeRobotを活用したロボット学習パイプラインの実装記録です。

## Overview

Physical AIに関する学習・実装を目的として、ロボットのデータ収集からデータ変換、模倣学習、推論までの一連の処理に取り組みました。

## Project Structure

```text
.
├── configs/
│   ├── conversion_config.yaml
│   ├── crane_features.py
│   ├── diff_drive_features.py
│   └── train_config.yaml
├── scripts/
│   ├── rosbag2lerobot.py
│   ├── train.py
│   ├── infer.py
│   ├── infer_diff_drive.py
│   ├── debug_policy.py
│   └── visualize_dataset.py
├── src/
├── lerobot/
├── pyproject.toml
└── pixi.lock
```

## Implementation

### 1. Data Conversion

- rosbag2からLeRobot形式へのデータ変換
- ロボットの特徴量定義
- データセットの構築

### 2. Dataset Visualization

- 変換したデータセットの可視化
- データの確認・検証

### 3. Imitation Learning

- Diffusion Policyの学習
- 学習パラメータの設定

### 4. Inference

- 学習済みPolicyによる推論
- ロボット制御の検証
- Policyのデバッグ

## Environment

- Ubuntu
- Python
- ROS 2
- Pixi
- LeRobot

## Setup

```bash
pixi install
```

## Usage

各処理の実行方法は `pyproject.toml` のPixiタスク定義を参照してください。

```bash
pixi task list
```

## Repository

LeRobot is managed as a Git submodule.

https://github.com/huggingface/lerobot

## License

See the licenses of the respective dependencies and source projects.

