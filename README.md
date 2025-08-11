# 強化学習による倒立振り子制御

## 概要

このプロジェクトは、強化学習アルゴリズム「DQN (Deep Q-Network)」を用いて、倒立振り子（CartPole）を制御するエージェントを開発するものです。
すべての開発と学習は、単一の環境（ローカルPCやブラウザ版VS Code）で完結します。

- **技術スタック:** Python, PyTorch, Gymnasium, Git

---

## 環境構築

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/naokimaron03-gif/PoleCart.git
    cd PoleCart
    ```

2.  **依存ライブラリのインストール:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 使い方

以下のコマンドで学習を開始します。
ブラウザ版VS Codeの場合は、ターミナルを開いてコマンドを実行してください。

```bash
python train.py