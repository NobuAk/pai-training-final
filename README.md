# Physical AI Training

ROS 2、rosbag2、LeRobotを活用したロボット学習パイプラインの実装記録です。

## 概要

Physical AIに関する学習と実装を目的として、ロボットのデータ収集からデータ変換、模倣学習、推論までの一連の処理に取り組みました。

本プロジェクトでは、ROS 2で収集したロボットの動作データをrosbag2からLeRobot形式に変換し、Diffusion Policyによる模倣学習と、学習済みモデルを用いた推論・ロボット制御の検証を行います。

## ディレクトリ構成

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

## 実装内容

### 1. データ変換

- rosbag2で記録したデータをLeRobot形式に変換
- ロボットの関節・センサ情報などの特徴量定義
- 模倣学習に利用するデータセットの構築

### 2. データセットの可視化

- 変換したデータセットの可視化
- 収集データの確認と検証
- 学習に使用するデータの品質確認

### 3. 模倣学習

- Diffusion Policyによる模倣学習
- 学習パラメータの設定
- ロボットの動作データを用いたPolicyの学習

### 4. 推論・ロボット制御

- 学習済みPolicyによる推論
- 推論結果を用いたロボット制御の検証
- Policyの動作確認・デバッグ

## 使用技術・開発環境

- Ubuntu
- Python
- ROS 2
- rosbag2
- Pixi
- LeRobot
- Diffusion Policy

## 環境構築

Pixiを使用して開発環境を構築します。

```bash
pixi install
```

## 実行方法

利用可能なタスクの一覧は、以下のコマンドで確認できます。

```bash
pixi task list
```

各処理の実行方法は `pyproject.toml` に定義されています。

## 外部ライブラリ

本プロジェクトでは、ロボット学習フレームワークとしてHugging FaceのLeRobotを使用しています。

- [LeRobot - GitHub](https://github.com/huggingface/lerobot)

LeRobotはGit Submoduleとして管理しています。

## ライセンス

本プロジェクトで使用している外部ライブラリおよび関連コードについては、それぞれのライセンスと利用条件に従います。
