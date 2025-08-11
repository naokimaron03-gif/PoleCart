import matplotlib.pyplot as plt
import os

def plot_rewards(rewards, filepath):
    """
    エピソードごとの合計報酬のリストを受け取り、グラフをプロットして画像として保存する。

    Args:
        rewards (list): 各エピソードの合計報酬を格納したリスト。
        filepath (str): 保存先のファイルパス (例: 'results/reward_plot.png')。
    """
    # 保存先ディレクトリが存在しない場合は作成
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(filepath)
    plt.close() # メモリリークを防ぐためにプロットを閉じる
    print(f"Graph saved to {filepath}")