import gymnasium as gym
import argparse
from agent import DQNAgent
from utils import plot_rewards
import os

def main(episodes, batch_size, learning_rate):
    """
    強化学習の学習プロセスを実行するメイン関数。
    """
    # --- 環境とエージェントの初期化 ---
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim, 
        action_dim,
        learning_rate=learning_rate
    )

    # --- 保存先パスの定義 ---
    model_path = 'models/dqn_cartpole.pth'
    plot_path = 'results/reward_plot.png'

    # --- 変数の初期化 ---
    all_rewards = []
    update_target_every = 10 # 10エピソード毎にターゲットネットワークを更新

    print("学習を開始します...")
    # --- 学習ループ ---
    for e in range(episodes):
        state, _ = env.reset() # 環境をリセット
        total_reward = 0
        done = False

        while not done:
            # エージェントに行動を選択させる
            action = agent.act(state)
            
            # 選択した行動を環境で実行
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # エピソード終了条件
            
            total_reward += reward

            # 経験をリプレイバッファに記憶させる
            agent.remember(state, action, reward, next_state, done)
            
            # 状態を更新
            state = next_state
            
            # リプレイバッファに十分な経験が溜まったら学習（replay）を実行
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        all_rewards.append(total_reward)
        print(f"エピソード: {e+1}/{episodes}, 合計報酬: {total_reward:.2f}, イプシロン: {agent.epsilon:.3f}")

        # 定期的にターゲットネットワークを更新
        if (e + 1) % update_target_every == 0:
            agent.update_target_network()
            # 途中経過を保存
            agent.save_model(model_path)


    env.close()
    
    # --- 最終結果の保存 ---
    print("\n学習が完了しました。")
    plot_rewards(all_rewards, plot_path)
    agent.save_model(model_path)

if __name__ == "__main__":
    # コマンドラインから引数を受け取るためのパーサーを設定
    parser = argparse.ArgumentParser(description='DQN agent for CartPole')
    parser.add_argument('--episodes', type=int, default=300, help='学習するエピソード数')
    parser.add_argument('--batch_size', type=int, default=64, help='学習時のバッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率 (learning rate)')
    args = parser.parse_args()

    # main関数を実行
    main(episodes=args.episodes, batch_size=args.batch_size, learning_rate=args.lr)