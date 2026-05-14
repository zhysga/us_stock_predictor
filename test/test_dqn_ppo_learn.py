import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

# 设置随机种子，确保实验可重现
# 通过设置随机种子，我们可以确保每次运行代码时得到相同的结果
# 这对于调试和比较不同算法的性能非常重要
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ==================== 股票交易环境 ====================
# 定义股票交易环境类，继承自object
# 这个环境模拟了一个简单的股票交易场景，智能体可以执行买入、卖出或持有操作
class StockTradingEnv:
    def __init__(self, data, initial_balance=10000):
        """
        初始化股票交易环境
        
        Args:
            data: 股票价格数据，numpy数组格式
            initial_balance: 初始资金，默认为10000
        """
        # 存储股票价格数据
        self.data = data
        # 设置初始资金
        self.initial_balance = initial_balance
        # 重置环境到初始状态
        self.reset()
        
    def reset(self):
        """
        重置环境状态到初始值
        
        Returns:
            初始状态观测值，包含价格、持股、余额等信息
        """
        # 当前时间步
        self.current_step = 0
        # 当前账户余额
        self.balance = self.initial_balance
        # 当前持有的股票数量
        self.shares_held = 0
        # 总卖出股票数量
        self.total_shares_sold = 0
        # 总卖出价值
        self.total_sales_value = 0
        # 历史最大净值
        self.max_net_worth = self.initial_balance
        # 返回初始观测状态
        return self._get_obs()
    
    def _get_obs(self):
        """
        获取当前时间步的观测状态
        状态向量包含8个特征：
        1. 归一化的当前价格
        2. 归一化的持有股份
        3. 余额占初始资金的比例
        4. 价格变化率
        5. 价格相对于移动平均的偏差
        6. 价格波动率
        7. 净值占初始资金的比例
        8. 时间进度
        
        Returns:
            obs: 状态观测值数组，形状为(8,)
        """
        # 边界检查：如果已经到达数据末尾，返回零向量
        if self.current_step >= len(self.data) - 1:
            return np.zeros(8)
            
        # 获取当前价格
        price = self.data[self.current_step]
        
        # 计算技术指标
        # 窗口大小：最多考虑10天的数据，但如果当前步数不足10天，则使用实际可用天数
        window = min(10, self.current_step + 1)
        # 获取最近window天的价格数据
        recent_prices = self.data[max(0, self.current_step - window + 1):self.current_step + 1]
        
        # 计算移动平均价格
        price_ma = np.mean(recent_prices)
        # 计算价格标准差（波动率）
        price_std = np.std(recent_prices) if len(recent_prices) > 1 else 0
        # 计算价格变化率
        price_change = (price - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0
        
        # 计算当前净值（余额+持股价值）
        net_worth = self.balance + self.shares_held * price
        
        # 构建状态观测向量
        obs = np.array([
            price / 1000,  # 价格归一化，避免数值过大
            self.shares_held / 1000,  # 持股归一化
            self.balance / self.initial_balance,  # 余额比例，表示资金使用情况
            price_change,  # 价格变化率，反映短期趋势
            (price - price_ma) / (price_ma + 1e-8),  # 价格相对于移动平均的偏差，用于判断超买超卖
            price_std / (price + 1e-8),  # 价格波动率，衡量风险
            net_worth / self.initial_balance,  # 净值比例，反映投资收益
            min(self.current_step / len(self.data), 1)  # 时间进度，帮助智能体了解交易周期
        ])
        
        # 转换为32位浮点数以节省内存
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        执行动作并更新环境状态
        
        Args:
            action: 执行的动作 (0=持有, 1=买入, 2=卖出)
            
        Returns:
            next_state: 下一状态观测值
            reward: 奖励值
            done: 是否结束
            info: 其他信息
        """
        # 边界检查：如果已经到达数据末尾，返回终止状态
        if self.current_step >= len(self.data) - 1:
            return self._get_obs(), 0, True, {}
            
        # 获取当前价格
        current_price = self.data[self.current_step]
        
        # 根据动作执行交易
        # 动作: 0=持有, 1=买入, 2=卖出
        if action == 1:  # 买入操作
            # 限制单次买入量为最多10股或账户余额可购买的数量，取较小值
            shares_to_buy = min(self.balance // current_price, 10)
            if shares_to_buy > 0:
                # 扣除购买股票的费用
                self.balance -= shares_to_buy * current_price
                # 增加持股数量
                self.shares_held += shares_to_buy
                
        elif action == 2:  # 卖出操作
            # 限制单次卖出量为最多10股或当前持股数量，取较小值
            shares_to_sell = min(self.shares_held, 10)
            if shares_to_sell > 0:
                # 增加卖出股票获得的资金
                self.balance += shares_to_sell * current_price
                # 减少持股数量
                self.shares_held -= shares_to_sell
                # 更新总卖出股数
                self.total_shares_sold += shares_to_sell
                # 更新总卖出价值
                self.total_sales_value += shares_to_sell * current_price
        
        # 时间步进1
        self.current_step += 1
        
        # 计算奖励
        # 当前净值 = 余额 + 持股价值
        net_worth = self.balance + self.shares_held * current_price
        
        # 奖励公式: 当前净值与历史最大净值的差值归一化
        # 这个奖励函数鼓励智能体创造新的净值高点
        # \[ R_t = \frac{W_t - W_{max}}{W_0} \]
        # 其中 \( W_t \) 是当前净值，\( W_{max} \) 是历史最大净值，\( W_0 \) 是初始资金
        reward = (net_worth - self.max_net_worth) / self.initial_balance
        
        # 如果创造了新的净值高点，给予额外奖励
        if net_worth > self.max_net_worth:
            self.max_net_worth = net_worth
            reward += 0.01  # 额外奖励创新高
            
        # 惩罚过度交易，避免智能体频繁买卖产生过多交易费用
        if action != 0:
            reward -= 0.001
            
        # 判断是否结束：已到达数据末尾
        done = self.current_step >= len(self.data) - 1
        
        # 返回下一状态、奖励、是否结束标志和其他信息
        return self._get_obs(), reward, done, {'net_worth': net_worth}

# ==================== 生成股票数据 ====================
def generate_stock_data(n_days=1000, initial_price=100):
    """
    生成模拟股票价格数据
    
    使用几何布朗运动(GBM)模型生成价格数据，这是一种常用的金融模型
    用于模拟股票价格的随机行为:
    \[ \Delta S = \mu S \Delta t + \sigma S \epsilon \sqrt{\Delta t} \]
    其中 \( S \) 是股票价格，\( \mu \) 是漂移率（期望收益率），
    \( \sigma \) 是波动率，\( \epsilon \) 是标准正态分布随机变量
    \( \Delta t \) 是时间步长
    
    Args:
        n_days: 交易天数，默认1000天
        initial_price: 初始价格，默认100元
        
    Returns:
        prices: 生成的股票价格序列，numpy数组格式
    """
    # 初始化价格序列
    prices = [initial_price]
    
    # 逐日生成价格数据
    for i in range(n_days - 1):
        # 设置模型参数
        dt = 1/252  # 日频率（一年252个交易日）
        mu = 0.05   # 年化收益率5%
        sigma = 0.2  # 年化波动率20%
        
        # 计算漂移项: \( \mu \Delta t \)
        # 漂移项代表价格的长期趋势
        drift = mu * dt
        
        # 计算随机项: \( \sigma \epsilon \sqrt{\Delta t} \)
        # 随机项代表价格的短期波动
        shock = sigma * np.sqrt(dt) * np.random.normal()
        
        # 添加一些周期性和趋势成分，使数据更接近真实市场
        # 周期项：模拟市场中常见的周期性波动
        cycle = 0.02 * np.sin(2 * np.pi * i / 50)  # 50天周期
        # 趋势项：模拟市场的长期上升趋势
        trend = 0.0001 * i  # 轻微上升趋势
        
        # 计算价格变化
        price_change = prices[-1] * (drift + shock + cycle + trend)
        # 计算新价格，确保价格不低于10元
        new_price = max(prices[-1] + price_change, 10)
        prices.append(new_price)
    
    # 返回numpy数组格式的价格序列
    return np.array(prices)

# ==================== DQN实现 ====================
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        DQN网络结构定义
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间维度
            hidden_size: 隐藏层大小
        """
        # 调用父类构造函数
        super(DQNNetwork, self).__init__()
        # 第一个全连接层：状态->隐藏层
        self.fc1 = nn.Linear(state_size, hidden_size)
        # 第二个全连接层：隐藏层->隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 第三个全连接层：隐藏层->动作价值
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态，形状为(batch_size, state_size)
            
        Returns:
            Q值输出，形状为(batch_size, action_size)
        """
        # 第一层：线性变换+ReLU激活函数
        x = F.relu(self.fc1(x))
        # 第二层：线性变换+ReLU激活函数
        x = F.relu(self.fc2(x))
        # 第三层：线性变换（输出Q值）
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        """
        DQN智能体初始化
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间维度
            lr: 学习率，默认0.001
        """
        # 存储状态空间和动作空间维度
        self.state_size = state_size
        self.action_size = action_size
        
        # 经验回FFER区，最大容量10000
        # 经验回放用于打破数据相关性，提高训练稳定性
        self.memory = deque(maxlen=10000)
        
        # ε-贪婪策略参数
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子
        
        # 创建主网络和目标网络
        # 主网络用于选择动作和训练
        # 目标网络用于计算目标Q值，定期与主网络同步
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        
        # 优化器：使用Adam优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 同步目标网络参数
        self.update_target_network()
        
    def update_target_network(self):
        """
        更新目标网络参数，使其与主网络同步
        这种延迟更新机制有助于训练稳定
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到经验回FFER区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        # 将经验元组添加到缓冲区
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根据当前策略选择动作
        
        使用ε-贪婪策略平衡探索与利用:
        \[ a = \begin{cases} 
        \text{random action} & \text{with probability } \epsilon \\
        \arg\max_a Q(s, a) & \text{with probability } 1-\epsilon
        \end{cases} \]
        
        Args:
            state: 当前状态
            
        Returns:
            action: 选择的动作
        """
        # ε-贪婪策略：以ε概率随机选择动作（探索）
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # 以(1-ε)概率选择最优动作（利用）
        # 将状态转换为PyTorch张量并增加批次维度
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # 通过主网络计算Q值
        q_values = self.q_network(state_tensor)
        # 选择Q值最大的动作
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """
        从经验回放中采样并训练网络
        
        Args:
            batch_size: 批次大小，默认32
        """
        # 如果经验不足一个批次，不进行训练
        if len(self.memory) < batch_size:
            return
        
        # 从经验回放中随机采样一个批次
        batch = random.sample(self.memory, batch_size)
        # 分别提取状态、动作、奖励、下一状态和终止标志
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # 计算当前Q值: \( Q(s_t, a_t) \)
        # gather函数根据动作索引提取对应的Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值: \( r_{t+1} + \gamma \max_{a} Q'(s_{t+1}, a) \)
        # 其中 \( Q' \) 是目标网络，\( \gamma \) 是折扣因子
        # 目标网络的使用有助于训练稳定
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        # 对于终止状态，目标Q值等于奖励；对于非终止状态，目标Q值等于奖励+折扣未来价值
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        # 计算损失函数: \( L = \mathbb{E}[(r_{t+1} + \gamma \max_{a} Q'(s_{t+1}, a) - Q(s_t, a_t))^2] \)
        # 使用均方误差损失函数
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==================== PPO实现 ====================
class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        PPO网络结构定义
        PPO使用共享网络架构，同时输出动作概率和状态价值
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间维度
            hidden_size: 隐藏层大小
        """
        # 调用父类构造函数
        super(PPONetwork, self).__init__()
        
        # 共享层：状态->隐藏特征
        # 使用Sequential容器按顺序定义网络层
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # 策略网络（Actor）：隐藏特征->动作概率
        self.actor = nn.Linear(hidden_size, action_size)
        
        # 价值网络（Critic）：隐藏特征->状态价值
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            shared_out: 共享层输出
        """
        # 通过共享层计算隐藏特征
        shared_out = self.shared(x)
        return shared_out
        
    def act(self, x):
        """
        策略网络前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            action_probs: 动作概率分布
        """
        # 先通过共享层
        shared_out = self.forward(x)
        # 再通过策略网络，并使用softmax获得动作概率分布
        action_probs = F.softmax(self.actor(shared_out), dim=-1)
        return action_probs
    
    def evaluate(self, x):
        """
        评估网络前向传播
        同时计算动作概率和状态价值
        
        Args:
            x: 输入状态
            
        Returns:
            action_probs: 动作概率分布
            state_value: 状态价值
        """
        # 通过共享层
        shared_out = self.forward(x)
        # 计算动作概率分布
        action_probs = F.softmax(self.actor(shared_out), dim=-1)
        # 计算状态价值
        state_value = self.critic(shared_out)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.003):
        """
        PPO智能体初始化
        
        Args:
            state_size: 状态空间维度
            action_size: 动作空间维度
            lr: 学习率，默认0.003
        """
        # 存储状态空间和动作空间维度
        self.state_size = state_size
        self.action_size = action_size
        
        # 创建PPO网络
        self.network = PPONetwork(state_size, action_size)
        
        # 优化器：使用Adam优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO超参数
        self.gamma = 0.95  # 折扣因子，衡量未来奖励的重要性
        self.epsilon_clip = 0.2  # PPO裁剪参数，限制策略更新幅度
        self.ppo_epochs = 4  # PPO更新轮数，每轮使用同一批数据多次训练
        
        # 存储轨迹数据用于训练
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.state_values = []
        self.is_terminals = []
        
    def act(self, state):
        """
        根据当前策略选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            action: 选择的动作
        """
        # 将状态转换为PyTorch张量并增加批次维度
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # 通过策略网络计算动作概率分布
        action_probs = self.network.act(state_tensor)
        
        # 修复：确保动作概率分布有效，避免NaN值
        # 添加小的常数防止log(0)和数值不稳定
        action_probs = torch.clamp(action_probs, 1e-10, 1.0)
        # 确保概率分布归一化
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        # 从概率分布中采样动作
        # 使用Categorical分布来处理离散动作空间
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # 存储轨迹数据用于后续训练
        self.states.append(state)
        self.actions.append(action.item())
        self.logprobs.append(dist.log_prob(action).item())
        
        return action.item()
    
    def store_reward(self, reward, is_terminal):
        """
        存储奖励和终止状态
        
        Args:
            reward: 获得的奖励
            is_terminal: 是否结束
        """
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
    
    def update(self):
        """
        更新策略网络和价值网络
        使用PPO算法进行策略优化
        """
        # 如果没有轨迹数据，不进行更新
        if len(self.states) == 0:
            return
            
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        # 逆向遍历奖励序列计算折扣奖励
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # 折扣奖励计算: \( R_t = r_t + \gamma R_{t+1} \)
            # 这个计算将未来奖励折现到当前时间步
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 转换为PyTorch张量
        rewards = torch.FloatTensor(rewards)
        # 奖励标准化: \( \hat{R}_t = \frac{R_t - \mu_R}{\sigma_R} \)
        # 标准化有助于提高训练稳定性
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        
        # 转换轨迹数据为PyTorch张量
        old_states = torch.FloatTensor(self.states)
        old_actions = torch.LongTensor(self.actions)
        old_logprobs = torch.FloatTensor(self.logprobs)
        
        # PPO更新
        # 多轮更新使用同一批数据，这是PPO的特点之一
        for _ in range(self.ppo_epochs):
            # 评估当前策略，计算动作概率和状态价值
            action_probs, state_values = self.network.evaluate(old_states)
            
            # 修复：确保动作概率分布有效，避免NaN值
            # 添加小的常数防止log(0)和数值不稳定
            action_probs = torch.clamp(action_probs, 1e-10, 1.0)
            # 确保概率分布归一化
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
            # 计算新旧策略的概率比
            # \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \)
            dist = torch.distributions.Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            # 概率比 = exp(新策略对数概率 - 旧策略对数概率)
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # 计算优势函数: \( A_t = R_t - V(s_t) \)
            # 优势函数衡量动作相对于平均表现的好坏
            advantages = rewards - state_values.squeeze()
            
            # PPO损失函数:
            # \[ L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)] \]
            # 裁剪机制防止策略更新过大
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失: \( L^{VF} = \frac{1}{2}(V(s_t) - R_t)^2 \)
            # 价值网络用于估计状态价值，帮助计算优势函数
            critic_loss = F.mse_loss(state_values.squeeze(), rewards)
            
            # 总损失: \( L = L^{CLIP} + c_1 L^{VF} \) (这里 \( c_1 = 0.5 \))
            # 同时优化策略网络和价值网络
            total_loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播更新网络参数
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # 清空存储的轨迹数据，为下一轮收集数据做准备
        self.clear_memory()
    
    def clear_memory(self):
        """
        清空存储的轨迹数据
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.state_values.clear()
        self.is_terminals.clear()

# ==================== 训练函数 ====================
def train_dqn(env, episodes=1000):
    """
    训练DQN智能体
    
    Args:
        env: 环境
        episodes: 训练轮数
        
    Returns:
        agent: 训练好的DQN智能体
        scores: 训练得分
    """
    # 获取状态空间维度
    state_size = len(env.reset())
    # 动作空间维度（持有、买入、卖出）
    action_size = 3
    
    # 创建DQN智能体
    agent = DQNAgent(state_size, action_size)
    # 存储训练得分
    scores = []
    
    # 开始训练循环
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        # 初始化本轮总奖励
        total_reward = 0
        
        # 一个episode的交互循环
        while True:
            # 根据当前策略选择动作
            action = agent.act(state)
            # 执行动作，观察环境反馈
            next_state, reward, done, info = env.step(action)
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            
            # 如果episode结束，跳出循环
            if done:
                break
        
        # 使用经验回放训练网络
        agent.replay()
        # 记录本轮得分
        scores.append(total_reward)
        
        # 定期更新目标网络
        if episode % 10 == 0:
            agent.update_target_network()
        
        # 定期打印训练进度
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"DQN Episode {episode}, Average Score: {avg_score:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    return agent, scores

def train_ppo(env, episodes=1000):
    """
    训练PPO智能体
    
    Args:
        env: 环境
        episodes: 训练轮数
        
    Returns:
        agent: 训练好的PPO智能体
        scores: 训练得分
    """
    # 获取状态空间维度
    state_size = len(env.reset())
    # 动作空间维度（持有、买入、卖出）
    action_size = 3
    
    # 创建PPO智能体
    agent = PPOAgent(state_size, action_size)
    # 存储训练得分
    scores = []
    
    # 开始训练循环
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        # 初始化本轮总奖励
        total_reward = 0
        
        # 一个episode的交互循环
        while True:
            # 根据当前策略选择动作
            action = agent.act(state)
            # 执行动作，观察环境反馈
            next_state, reward, done, info = env.step(action)
            # 存储奖励和终止状态
            agent.store_reward(reward, done)
            
            # 更新状态和累计奖励
            state = next_state
            total_reward += reward
            
            # 如果episode结束，跳出循环
            if done:
                break
        
        # 更新策略
        agent.update()
        # 记录本轮得分
        scores.append(total_reward)
        
        # 定期打印训练进度
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"PPO Episode {episode}, Average Score: {avg_score:.4f}")
    
    return agent, scores

# ==================== 测试和评估 ====================
def test_agent(env, agent, agent_type="DQN"):
    """
    测试智能体性能
    
    Args:
        env: 环境
        agent: 智能体
        agent_type: 智能体类型
        
    Returns:
        total_reward: 总奖励
        actions_taken: 执行的动作序列
        net_worths: 净值序列
    """
    # 重置环境
    state = env.reset()
    # 初始化总奖励
    total_reward = 0
    # 存储动作序列
    actions_taken = []
    # 存储净值序列
    net_worths = [env.initial_balance]
    
    # 测试循环
    while True:
        # 根据智能体类型选择相应的动作函数
        if agent_type == "DQN":
            action = agent.act(state)
        else:  # PPO
            action = agent.act(state)
            
        # 记录动作
        actions_taken.append(action)
        # 执行动作
        next_state, reward, done, info = env.step(action)
        # 累计奖励
        total_reward += reward
        # 记录净值
        net_worths.append(info['net_worth'])
        
        # 更新状态
        state = next_state
        # 如果结束，跳出循环
        if done:
            break
    
    return total_reward, actions_taken, net_worths

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 生成股票数据
    # 使用默认参数生成1000天的股票价格数据，初始价格为100元
    print("生成股票数据...")
    stock_data = generate_stock_data(n_days=1000, initial_price=100)
    
    # 创建环境
    # 使用生成的股票数据初始化交易环境，初始资金为10000元
    env = StockTradingEnv(stock_data)
    
    # 训练DQN智能体
    # 设置训练轮数为500轮，每轮包含完整的交易周期
    print("开始训练DQN...")
    dqn_agent, dqn_scores = train_dqn(env, episodes=500)
    
    # 训练PPO智能体
    # 设置训练轮数为500轮，每轮包含完整的交易周期
    print("开始训练PPO...")
    ppo_agent, ppo_scores = train_ppo(env, episodes=500)
    
    # 测试智能体性能
    # 在相同的股票数据上测试训练好的智能体
    print("\n测试结果:")
    
    # 测试DQN智能体
    # 重置环境以确保测试在相同条件下进行
    env.reset()
    # 调用测试函数评估DQN智能体性能
    dqn_reward, dqn_actions, dqn_net_worths = test_agent(env, dqn_agent, "DQN")
    # 打印DQN测试结果：总回报和最终净值
    print(f"DQN - 总回报: {dqn_reward:.4f}, 最终净值: {dqn_net_worths[-1]:.2f}")
    
    # 测试PPO智能体
    # 重置环境以确保测试在相同条件下进行
    env.reset()
    # 调用测试函数评估PPO智能体性能
    ppo_reward, ppo_actions, ppo_net_worths = test_agent(env, ppo_agent, "PPO")
    # 打印PPO测试结果：总回报和最终净值
    print(f"PPO - 总回报: {ppo_reward:.4f}, 最终净值: {ppo_net_worths[-1]:.2f}")
    
    # 基准比较（买入并持有策略）
    # 实现一个简单的买入并持有策略作为基准
    env.reset()
    # 初始净值等于初始资金
    benchmark_net_worth = env.initial_balance
    # 获取初始价格和最终价格
    initial_price = stock_data[0]
    final_price = stock_data[-1]
    # 计算可购买的股票数量（整数股）
    shares_bought = env.initial_balance // initial_price
    # 计算最终净值：股票价值 + 剩余现金
    benchmark_final = shares_bought * final_price + (env.initial_balance % initial_price)
    # 计算基准策略回报率
    benchmark_return = (benchmark_final - env.initial_balance) / env.initial_balance
    
    # 打印基准策略结果：最终净值和回报率
    print(f"买入持有基准 - 最终净值: {benchmark_final:.2f}, 回报率: {benchmark_return:.4f}")
    
    # 绘制结果图表
    # 创建一个15x10英寸的图形窗口
    plt.figure(figsize=(15, 10))
    
    # 训练曲线子图
    # 显示DQN和PPO在训练过程中的得分变化
    plt.subplot(2, 3, 1)
    plt.plot(dqn_scores, label='DQN', alpha=0.7)
    plt.plot(ppo_scores, label='PPO', alpha=0.7)
    plt.title('训练曲线')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # 股票价格子图
    # 显示生成的股票价格序列
    plt.subplot(2, 3, 2)
    plt.plot(stock_data)
    plt.title('股票价格')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.grid(True)
    
    # 净值比较子图
    # 对比DQN、PPO和基准策略的净值变化
    plt.subplot(2, 3, 3)
    plt.plot(dqn_net_worths, label='DQN', linewidth=2)
    plt.plot(ppo_net_worths, label='PPO', linewidth=2)
    plt.axhline(y=benchmark_final, color='r', linestyle='--', label='Buy & Hold')
    plt.title('净值对比')
    plt.xlabel('Day')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.grid(True)
    
    # DQN动作分布子图
    # 显示DQN智能体选择各动作的次数
    plt.subplot(2, 3, 4)
    action_names = ['Hold', 'Buy', 'Sell']
    dqn_action_counts = [dqn_actions.count(i) for i in range(3)]
    plt.bar(action_names, dqn_action_counts)
    plt.title('DQN动作分布')
    plt.ylabel('Count')
    
    # PPO动作分布子图
    # 显示PPO智能体选择各动作的次数
    plt.subplot(2, 3, 5)
    ppo_action_counts = [ppo_actions.count(i) for i in range(3)]
    plt.bar(action_names, ppo_action_counts)
    plt.title('PPO动作分布')
    plt.ylabel('Count')
    
    # 收益对比子图
    # 对比三种策略的最终回报率
    plt.subplot(2, 3, 6)
    returns = [(dqn_net_worths[-1] - env.initial_balance) / env.initial_balance,
              (ppo_net_worths[-1] - env.initial_balance) / env.initial_balance,
              benchmark_return]
    plt.bar(['DQN', 'PPO', 'Buy & Hold'], returns)
    plt.title('收益率对比')
    plt.ylabel('Return Rate')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 自动调整子图间距
    plt.tight_layout()
    # 显示图形
    plt.show()
    
    # 算法特点总结
    # 打印DQN和PPO算法的特点和优势
    print("\n算法特点总结:")
    print("DQN优势: 样本效率高，适合离散动作空间，有经验回放机制")
    print("PPO优势: 训练稳定，适合连续动作空间，策略更新更平滑")
    print("实际应用中，PPO通常在复杂环境中表现更稳定")