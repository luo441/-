import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import torch
import matplotlib
print(matplotlib.matplotlib_fname())  # 输出配置文件路径，例如：/usr/local/lib/python3.9/site-packages/matplotlib/mpl-data/matplotlibrc
# 创建Frozen Lake环境
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")

# -------------------- CUDA加速配置 --------------------
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化器
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Q-Learning参数
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 2000
EXPLORATION_RATE = 1.0
MAX_EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY = 0.001

# 初始化Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# 训练统计
episode_rewards = []
episode_exploration = []
success_rate = []

# Q-Learning训练过程
print("开始Q-Learning训练...")
start_time = time.time()

for episode in range(EPISODES):
    # 显示训练进度
    if episode % SHOW_EVERY == 0:
        print(f"训练进度: {episode}/{EPISODES} ({(episode/EPISODES)*100:.1f}%)")
    
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # 探索-利用策略
        if np.random.random() < EXPLORATION_RATE:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[state])  # 利用
            
        # 执行动作
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 更新Q值 (Q-learning公式)
        q_table[state, action] = q_table[state, action] + LEARNING_RATE * (
            reward + DISCOUNT * np.max(q_table[new_state]) - q_table[state, action]
        )
        
        state = new_state
        total_reward += reward
        steps += 1
    
    # 记录训练统计
    episode_rewards.append(total_reward)
    episode_exploration.append(EXPLORATION_RATE)
    
    # 更新探索率（指数衰减）
    EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY * episode)
    
    # 计算最近100次成功率
    if episode >= 100:
        recent_success = np.mean(episode_rewards[-100:])
        success_rate.append(recent_success)

# 计算训练时间
training_time = time.time() - start_time
print(f"训练完成! 耗时: {training_time:.2f}秒")

# 训练结果可视化
# 设置支持中文的字体（根据系统选择）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(episode_rewards, label='每局奖励')
plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), label='100局平均奖励', color='r')
plt.xlabel('训练局数')
plt.ylabel('奖励')
plt.title('Q-Learning 训练进度')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(episode_exploration, label='探索率')
plt.plot(success_rate, label='成功率(100局平均)', color='g')
plt.xlabel('训练局数')
plt.legend()
plt.tight_layout()
plt.savefig('q_learning_training.png')
plt.show()

# 验证训练后的智能体
print("\n验证训练后的智能体：")
test_episodes = 10
total_test_reward = 0
success_count = 0

for episode in range(test_episodes):
    state, info = env.reset()
    done = False
    print(f"\n第 {episode+1} 次测试:")
    step_count = 0
    
    while not done:
        action = np.argmax(q_table[state])
        new_state, reward, done, truncated, info = env.step(action)
        total_test_reward += reward
        
        # 可视化智能体决策
        env_screen = env.render()
        plt.imshow(env_screen)
        plt.axis('off')
        plt.title(f"状态: {state}, 动作: {['左', '下', '右', '上'][action]}")
        plt.pause(0.5)
        plt.clf()
        
        state = new_state
        step_count += 1
        if done:
            if reward > 0:
                success_count += 1
                print(f"成功到达目标! 步数: {step_count}")
            else:
                print(f"掉入冰洞! 步数: {step_count}")
            break

env.close()

# 打印最终Q-table
print("\n学习到的Q-table (部分展示):")
print(q_table[:4])  # 只展示前4个状态

# 计算成功率
training_success_count = np.sum(np.array(episode_rewards) > 0)
print(f"\n训练结果: {training_success_count}/{EPISODES} 次成功 (成功率: {training_success_count/EPISODES*100:.2f}%)")
print(f"测试结果: {success_count}/{test_episodes} 次成功 (成功率: {success_count/test_episodes*100:.2f}%)")
print(f"测试平均奖励: {total_test_reward/test_episodes:.2f}")