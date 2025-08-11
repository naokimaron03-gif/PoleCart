import gymnasium as gym
import argparse
from agent import DQNAgent
from utils import plot_rewards
import os

# gymnasiumのラッパー（追加機能）をインポート
from gymnasium.wrappers import RecordVideo

def main(episodes, batch_size, learning_rate):
    """
    強化学習の学習プロセスを実行するメイン関数。
    """
    # --- 環境の初期化 ---
    # render_modeを'rgb_array'に設定。これは描画データをピクセル配列として受け取るための設定。
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # --- 動画保存ラッパーの設定 ---
    # envをRecordVideoでラップする。
    # 第2引数: 動画の保存先フォルダ
    # episode_trigger: どのエピソードを録画するかの条件を指定する関数。
    #                  ここでは、50エピソードごと（0, 50, 100, ...）に録画する設定。
    video_save_path = 'videos'
    env = RecordVideo(env, video_save_path, episode_trigger=lambda e: e % 50 == 0)


    # --- エージェントの初期化 ---
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
    update_target_every = 10 

    print("学習を開始します...")
    # --- 学習ループ ---
    for e in range(episodes):
        # env.reset()は動画ラッパーによって観測値と情報のタプルを返すようになる
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 内部でenv.render()が呼ばれ、録画中はフレームが記録される
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        all_rewards.append(total_reward)
        print(f"エピソード: {e+1}/{episodes}, 合計報酬: {total_reward:.2f}, イプシロン: {agent.epsilon:.3f}")

        if (e + 1) % update_target_every == 0:
            agent.update_target_network()
            agent.save_model(model_path)


    env.close()
    
    print("\n学習が完了しました。")
    plot_rewards(all_rewards, plot_path)
    agent.save_model(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent for CartPole')
    parser.add_argument('--episodes', type=int, default=300, help='学習するエピソード数')
    parser.add_argument('--batch_size', type=int, default=64, help='学習時のバッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率 (learning rate)')
    args = parser.parse_args()
    
    main(episodes=args.episodes, batch_size=args.batch_size, learning_rate=args.lr)