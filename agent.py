import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# Q-Networkの定義
class QNetwork(nn.Module):
    """
    状態を入力とし、各行動のQ値を出力するニューラルネットワーク。
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNエージェントの定義
class DQNAgent:
    """
    DQNアルゴリズムを実装したエージェント。
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000) # 経験を保存するリプレイバッファ
        self.gamma = gamma # 割引率
        self.epsilon = epsilon # ε-greedy法における探索の確率
        self.epsilon_decay = epsilon_decay # εの減衰率
        self.epsilon_min = epsilon_min # εの最小値
        
        # GPUが利用可能ならGPUを、そうでなければCPUを使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 2つのニューラルネットワークを初期化
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device) # 行動決定用
        self.target_net = QNetwork(state_dim, action_dim).to(self.device) # ターゲットQ値計算用
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 重みを同期
        self.target_net.eval() # ターゲットネットワークは評価モードに

        # オプティマイザと損失関数の設定
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        """
        ε-greedy法に基づいて行動を選択する。
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim) # 探索(Exploration): ランダムに行動
        
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return np.argmax(action_values.cpu().data.numpy()) # 活用(Exploitation): Q値が最大の行動

    def remember(self, state, action, reward, next_state, done):
        """
        経験（遷移）をリプレイバッファに保存する。
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        リプレイバッファからミニバッチをサンプリングし、ネットワークを学習する。
        """
        if len(self.memory) < batch_size:
            return # バッチサイズ分の経験が溜まっていなければ何もしない
        
        minibatch = random.sample(self.memory, batch_size)
        
        # 経験データを各変数に展開
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # テンソルに変換してデバイスに送る
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # 現在のQ値 (Q(s,a)) を計算
        q_values = self.policy_net(states).gather(1, actions)
        
        # ターゲットQ値 (r + γ * max_a' Q_target(s',a')) を計算
        # 次の状態が終端状態(done=True)の場合、将来の価値は0とする
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # 損失を計算してネットワークを更新
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilonを減衰させる
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """
        ターゲットネットワークの重みをポリシーネットワークの重みで上書きする。
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        """
        学習済みモデル（ポリシーネットワークの重み）を保存する。
        """
        # 保存先ディレクトリがなければ作成
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        保存されたモデルを読み込む。
        """
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # ターゲットネットも同期
        print(f"Model loaded from {filepath}")