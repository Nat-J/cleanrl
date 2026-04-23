# ==============================================================================
# PPO(Proximal Policy Optimization) 算法实现
# ==============================================================================
# 核心思想一图以蔽之：
#   1. 用当前策略与环境交互，收集一批数据 (obs, action, reward, done)
#   2. 用 GAE 计算每个时刻的"优势"——这个动作比平均水平好多少
#   3. 用这批数据更新策略网络，但限制更新幅度 (clipping)，防止"步子太大"
#   4. 重复上述过程
#
# 为什么需要 PPO？
#   - 传统策略梯度：一次更新后策略变了，之前收集的数据就废了 (on-policy)
#   - PPO 的 clipping：允许有限度地"重用"旧数据，提高样本效率
#
# 关键公式：
#   - 重要性采样比：ratio = π_new(a|s) / π_old(a|s)
#   - PPO 裁剪损失：L = min(ratio·A, clip(ratio,1-ε,1+ε)·A)
#   - GAE 优势估计：A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
# ==============================================================================
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
#
# PPO 算法整体架构流程图：
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                           PPO 训练主循环                                     │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │  1. 采样阶段 (Rollout)                                                      │
# │     环境 → 观测 → Agent → 动作 → 环境 → 奖励 → 存储到 buffer                   │
# │                                                                             │
# │  2. 优势计算 (GAE)                                                          │
# │     从后向前反向计算：δ_t = r_t + γV(s') - V(s)                              │
# │     累积优势：A_t = δ_t + γλA_{t+1}                                          │
# │                                                                             │
# │  3. 策略优化 (PPO-Clip)                                                     │
# │     将 buffer 打乱成 minibatch，多轮更新                                       │
# │     计算重要性采样比 r(θ) = π_new(a|s) / π_old(a|s)                          │
# │     裁剪损失：L^CLIP = min(r·A, clip(r,1-ε,1+ε)·A)                           │
# │                                                                             │
# │  4. 价值函数更新                                                             │
# │     损失：L^VF = (V(s) - R)^2                                                │
# │                                                                             │
# │  5. 熵正则化                                                                │
# │     鼓励探索：L^ent = -H(π)                                                   │
# │                                                                             │
# │  总损失：L = L^PG + c1·L^VF - c2·H(π)                                        │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# 关键数据结构（Buffer）：
#   obs[num_steps, num_envs, ...]    - 存储每一步的观测
#   actions[num_steps, num_envs, ...] - 存储每一步的动作
#   logprobs[num_steps, num_envs]    - 存储动作的对数概率（用于计算重要性采样比）
#   rewards[num_steps, num_envs]     - 存储每一步的奖励
#   dones[num_steps, num_envs]       - 存储每一步的终止标志
#   values[num_steps, num_envs]      - 存储每一步的状态价值（用于 GAE 和优势计算）
#
# PPO 核心思想：
#   - On-policy 算法：每次更新后丢弃旧数据，重新采样
#   - Clipping 机制：限制策略更新幅度，避免训练崩溃
#   - GAE 优势估计：在偏差和方差之间取得平衡

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # =========================================================================
    # 实验配置部分：控制实验跟踪、日志、视频录制等
    # =========================================================================
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # =========================================================================
    # PPO 算法超参数：这些参数直接影响算法行为和性能
    # =========================================================================
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments - 并行采样提高效率"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout - 每次迭代每环境的步数"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma - 未来奖励的折现率，越接近 1 越重视长期奖励"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation - GAE 的λ参数，控制偏差 - 方差权衡"""
    num_minibatches: int = 4
    """the number of mini-batches - 将 batch 分成多少份，每份用于一次梯度更新"""
    update_epochs: int = 4
    """the K epochs to update the policy - 每个数据轮用于更新的轮数"""
    norm_adv: bool = True
    """Toggles advantages normalization - 标准化优势有助于训练稳定"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient - PPO 的核心超参数，控制策略更新幅度"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy - 熵正则化系数，鼓励探索"""
    vf_coef: float = 0.5
    """coefficient of the value function - 价值函数损失的权重"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping - 防止梯度爆炸"""
    target_kl: float = None
    """the target KL divergence threshold - 可选的早停条件，防止策略更新过大"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime) - num_envs * num_steps"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime) - batch_size / num_minibatches"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime) - total_timesteps / batch_size"""


def make_env(env_id, idx, capture_video, run_name):
    # ========================================================================
    # 环境创建工厂函数
    # 为什么需要工厂函数？因为每个环境需要独立的配置（如视频录制只在 idx=0 时开启）
    # SyncVectorEnv 会并行执行多个环境，每次 step() 同时执行所有环境的 step
    # ========================================================================
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # ========================================================================
    # 权重初始化：正交初始化（Orthogonal Initialization）
    # 为什么需要特殊初始化？
    #   - 随机初始化权重有助于训练稳定性，特别是对于 tanh 激活函数
    #   - 价值函数输出层 std=1.0，actor 输出层 std=0.01 是经验值
    # ========================================================================
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    # ========================================================================
    # Agent 网络架构：Actor-Critic 结构
    #
    # ┌─────────────────────────────────────────────────────────────┐
    # │                      输入：观测状态 s                         │
    # └─────────────────────┬───────────────────────────────────────┘
    #                       │
    #          ┌────────────┴────────────┐
    #          │                         │
    #          ▼                         ▼
    # ┌─────────────────┐       ┌─────────────────┐
    # │   Actor 网络     │       │   Critic 网络    │
    # │ (策略函数π(a|s)) │       │  (价值函数 V(s))  │
    # └────────┬────────┘       └────────┬────────┘
    #          │                         │
    #          ▼                         ▼
    #   输出：动作概率分布           输出：状态价值
    #   (logits → softmax)           (标量值)
    #
    # 网络结构说明：
    #   - 共享相同的隐藏层结构（两层 64 单元 tanh）
    #   - Actor 输出：动作空间的 logits（离散动作）
    #   - Critic 输出：单一价值标量
    # ========================================================================
    def __init__(self, envs):
        super().__init__()
        # Critic 网络：估计状态价值 V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # 输出层 std=1.0 是价值函数的经验初始化
        )
        # Actor 网络：输出动作概率分布
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),  # 输出层 std=0.01 使初始策略接近均匀分布
        )

    def get_value(self, x):
        # 仅用于 GAE 计算时获取状态价值
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # ====================================================================
        # 核心方法：获取动作和相关训练信号
        # 这个方法在两个阶段被调用：
        #   1. 采样阶段 (no_grad): 采样动作执行于环境
        #   2. 更新阶段：重新计算 logprob 用于策略梯度
        #
        # 返回值说明：
        #   - action: 采样的动作（训练时可以是历史动作，用于计算 logprob）
        #   - log_prob: 动作的对数概率 π(a|s)，用于计算重要性采样比
        #   - entropy: 策略熵，用于鼓励探索的正则化项
        #   - value: 状态价值 V(s)，用于优势计算
        # ====================================================================
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



if __name__ == "__main__":
    # =========================================================================
    # 第一部分：初始化和配置
    # =========================================================================
    # 解析命令行参数（使用 tyro 库）
    args = tyro.cli(Args)
    # 根据配置计算运行时参数
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # =========================================================================
    # 环境设置：使用 SyncVectorEnv 并行执行多个环境
    # =========================================================================
    envs =  gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) #使用元组拼接，(a,b)+(c,0)，得到(a,b,c)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # ================================================================
        # 学习率退火 (Learning Rate Annealing)
        # ================================================================
        # 随着训练进行逐渐降低学习率，有助于后期稳定收敛
        # 公式：lr = lr_init × (1 - iteration / total_iterations)
        # ================================================================
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # TODO: 调用 agent.get_action_and_value() 获取当前步的动作和对数概率
            # 提示: action, logprob, _, value = agent.get_action_and_value(next_obs)
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            # TODO: 将 value 存储到 values 缓冲区中
            with torch.no_grad():
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ========================================================================
        # GAE(General Advantage Estimation) 优势计算
        # ========================================================================
        # 为什么需要 GAE？
        #   强化学习中，我们需要评估"某个动作比平均水平好多少"——这就是优势 A(s,a)
        #   直接 Monte Carlo: A = 实际回报 - 基线，方差高
        #   1-step TD: A = r + γV(s') - V(s)，偏差高
        #   GAE 通过λ参数在偏差和方差之间插值
        #
        # 计算顺序：从后向前 (反向)
        #   原因：A_t = δ_t + γλ·A_{t+1}，当前时刻的优势依赖未来时刻
        #
        # 关键变量解释：
        #   - nextnonterminal = 1 - done
        #     如果 episode 结束了 (done=1)，则下一步的价值不应该被计入
        #   - next_value: 最后一步的 bootstrap 值，处理未结束的 episode
        #
        # 公式：
        #   δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)   [TD 误差]
        #   A_t = δ_t + γ·λ·(1-done)·A_{t+1}             [GAE 累积]
        #   returns = A + V                               [优势 + 基线 = 目标回报]
        # ========================================================================
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    # TODO: 最后一步的 nextnonterminal = 1 - next_done (episode 是否未结束)
                    nextnonterminal = 1 - next_done
                    # TODO: 最后一步的 nextvalues = next_value (之前计算的 bootstrapped value)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # TODO: GAE 公式 δ_t = r_t + γ * V(s_{t+1}) * nextnonterminal - V(s_t)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # TODO: GAE 累积：A_t = δ_t + γ * λ * nextnonterminal * A_{t+1}
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # ================================================================
        # 数据展平 (Flatten the Batch)
        # ================================================================
        # 将 [num_steps, num_envs, ...] 展平为 [batch_size, ...]
        # batch_size = num_steps × num_envs
        #
        # 为什么要展平？
        #   - 之后需要随机打乱数据 (shuffle)，要求数据是连续的
        #   - 不同时间步、不同环境的数据可以混合在一起
        # ================================================================
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ========================================================================
        # PPO 优化阶段：使用采样数据更新策略和价值网络
        #
        # 为什么要打乱数据并使用 minibatch？
        #   - 打乱数据：打破时间序列相关性，使梯度估计更无偏
        #   - Minibatch：利用批量数据的统计特性，训练更稳定
        #
        # update_epochs: 同一批数据重复使用多次，提高数据效率
        #   - 注意：这是"off-policy"式重用，可能导致分布偏移
        #   - PPO 的 clipping 机制限制了这种偏移的上限
        # ========================================================================
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                # 从打乱的索引中取出当前 minibatch 的索引
                mb_inds = b_inds[start:end]

                # TODO: 用当前策略网络重新评估 minibatch 数据，获取新的 logprob、entropy 和 value
                # 提示: _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds])
                _, newlogprob, entropy, newvalue = ###
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    # TODO: 计算 KL 散度 approx_kl = (ratio - 1) - log(ratio)
                    approx_kl = ###
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                # ============================================================
                # 步骤 2：PPO 策略损失 (Policy Loss)
                # ============================================================
                # PPO 的核心创新：Clipping 机制
                #
                # 未裁剪损失：pg_loss1 = -A · ratio
                #   - 如果 A > 0 (好动作)：ratio 越大越好
                #   - 如果 A < 0 (坏动作)：ratio 越小越好
                #
                # 裁剪损失：pg_loss2 = -A · clip(ratio, 1-ε, 1+ε)
                #   - 限制 ratio 的变化范围在 [1-ε, 1+ε] 内
                #   - 防止策略更新过大导致训练崩溃
                #
                # 最终损失：取两者的最大值
                #   - 保证新策略不会偏离旧策略太多
                # ============================================================
                pg_loss1 = -mb_advantages * ratio
                # TODO: PPO 裁剪损失 pg_loss2 = -mb_advantages * clip(ratio, 1-clip_coef, 1+clip_coef)
                pg_loss2 = ###
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()


                # ============================================================
                # 步骤 3：价值函数损失 (Value Loss)
                # ============================================================
                # 价值函数的作用：估计状态价值 V(s)，用于 GAE 计算
                # 裁剪机制：同样限制价值函数的更新幅度，防止估计值波动太大
                # ============================================================
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # TODO: 未裁剪的价值损失 v_loss_unclipped = (newvalue - b_returns)^2
                    v_loss_unclipped = ###
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # TODO: 裁剪后的价值损失 v_loss_clipped = (v_clipped - b_returns)^2
                    v_loss_clipped = ###
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()


                # ============================================================
                # 步骤 4：熵正则化 (Entropy Regularization)
                # ============================================================
                # 熵的作用：度量策略的"确定性"
                #   - 高熵 = 策略接近均匀分布，探索性强
                #   - 低熵 = 策略集中在某些动作，利用性强
                #
                # 熵正则化：在损失中减去熵，鼓励探索
                #   - 防止策略过早收敛到局部最优
                # ============================================================
                entropy_loss = entropy.mean()
                # TODO: 总损失 = 策略损失 + 价值损失 * vf_coef - 熵 * ent_coef
                loss = ###

                # ============================================================
                # 步骤 5：梯度更新
                # ============================================================
                # 1. 清零梯度
                # 2. 反向传播计算梯度
                # 3. 梯度裁剪：防止梯度爆炸，稳定训练
                # 4. 优化器步进更新参数
                # ============================================================
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # target_kl 早停：如果 KL 散度太大，提前结束当前迭代
            # 这是一种安全机制，防止策略更新过大
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ================================================================
        # 评估指标：Explained Variance (解释方差)
        # ================================================================
        # 度量价值函数对回报的预测能力
        #   - explained_var = 1: 完美预测
        #   - explained_var = 0: 不比预测均值好
        #   - explained_var < 0: 比预测均值还差
        # ================================================================
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ================================================================
        # 日志记录：使用 TensorBoard 跟踪训练进度
        # ================================================================
        # 重要指标说明：
        #   - value_loss: 价值函数的拟合误差，应该逐渐下降
        #   - policy_loss: 策略损失，越低越好
        #   - entropy: 策略熵，初期高 (探索)，后期低 (利用)
        #   - approx_kl: 策略更新幅度，监控是否过大
        #   - clipfrac: 被裁剪的梯度比例，过高说明 clip_coef 太小
        #   - explained_variance: 价值函数预测质量
        #   - SPS: Samples Per Second，训练速度
        # ================================================================
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
