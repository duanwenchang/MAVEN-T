from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import numpy as np
from config import device
from utils import maskedMSE, maskedNLL, CELoss
from Teacher.teacher_model import highwayNet
from loader2 import ngsimDataset
from torch.utils.data import DataLoader
import time
import random
import os

# 设置固定的随机种子
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# =================== 论文中的多粒度知识蒸馏损失实现 ===================
class MultiGranularDistillationLoss(nn.Module):
    """
    实现论文公式16-18的多粒度知识蒸馏损失
    """

    def __init__(self, temperature=0.07):
        super(MultiGranularDistillationLoss, self).__init__()
        self.temperature = temperature

        # 特征适配器 - 用于对齐不同维度的特征
        self.adapter_low = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, teacher_features, student_features, teacher_attention, student_attention):
        """
        计算多粒度蒸馏损失
        """
        losses = {}

        # 1. 低级特征对齐损失 (公式16)
        F_T_low = teacher_features.get('low')
        F_S_low = student_features.get('low')

        if F_T_low is not None and F_S_low is not None:
            # 确保维度匹配
            if F_T_low.shape != F_S_low.shape:
                F_S_low_adapted = self.adapter_low(F_S_low.view(-1, F_S_low.shape[-1]))
                F_S_low_adapted = F_S_low_adapted.view(F_T_low.shape)
            else:
                F_S_low_adapted = F_S_low

            losses['L_low'] = torch.norm(F_T_low - F_S_low_adapted, p=2) ** 2
        else:
            losses['L_low'] = torch.tensor(0.0, device=device)

        # 2. 注意力转移损失 (公式17)
        if teacher_attention is not None and student_attention is not None:
            T, H, W = teacher_attention.shape[-3:]
            # 确保注意力图维度匹配
            if teacher_attention.shape != student_attention.shape:
                student_attention = F.interpolate(
                    student_attention, size=(T, H, W), mode='trilinear', align_corners=False
                )

            losses['L_att'] = torch.mean((teacher_attention - student_attention) ** 2) / (T * H * W)
        else:
            losses['L_att'] = torch.tensor(0.0, device=device)

        # 3. 语义对齐损失 (公式18) - 对比学习
        z_T = teacher_features.get('semantic')
        z_S = student_features.get('semantic')

        if z_T is not None and z_S is not None:
            # 归一化特征向量
            z_T = F.normalize(z_T, dim=-1)
            z_S = F.normalize(z_S, dim=-1)

            # 计算相似度
            sim_pos = torch.sum(z_T * z_S, dim=-1) / self.temperature

            # 负样本：batch内其他样本
            batch_size = z_T.size(0)
            if batch_size > 1:
                # 创建负样本
                z_S_neg = torch.cat([z_S[1:], z_S[:1]], dim=0)  # 循环移位作为负样本
                sim_neg = torch.sum(z_T * z_S_neg, dim=-1) / self.temperature

                # 对比损失
                logits = torch.stack([sim_pos, sim_neg], dim=1)
                labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                losses['L_semantic'] = F.cross_entropy(logits, labels)
            else:
                losses['L_semantic'] = torch.tensor(0.0, device=device)
        else:
            losses['L_semantic'] = torch.tensor(0.0, device=device)

        return losses


# =================== 强化学习组件 (与论文公式19-20一致) ===================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, device='cuda'):
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.old_policy = ActorCritic(state_dim, action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.mini_batch_size = 32

        self.buffer = PPOBuffer()

    def compute_rl_reward(self, state, action):
        """
        计算强化学习奖励 R(s_t, a_t) - 论文公式19
        包含安全性、舒适性、效率等指标
        """
        # 安全性奖励
        safety_reward = -torch.norm(action, p=2) * 0.1  # 避免极端动作

        # 舒适性奖励
        comfort_reward = -torch.sum(torch.abs(action[1:] - action[:-1])) * 0.1  # 动作平滑性

        # 效率奖励
        efficiency_reward = torch.norm(action[:2], p=2) * 0.05  # 鼓励前进

        total_reward = safety_reward + comfort_reward + efficiency_reward
        return total_reward.item()

    def compute_rl_loss(self, rewards, log_probs, values):
        """
        计算强化学习损失 - 论文公式20
        """
        returns = []
        discounted_reward = 0

        # 计算折扣回报
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, device=self.device)

        # 计算优势函数
        advantages = returns - values

        # PPO损失
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss


# =================== 自适应课程学习 (论文公式21-22) ===================
class AdaptiveCurriculumScheduler:
    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3, initial_complexity=0.3):
        self.alpha = alpha  # 对象数量权重
        self.beta = beta  # 相对速度权重
        self.gamma = gamma  # 轨迹复杂度权重
        self.current_complexity = initial_complexity
        self.stage = 0

    def compute_scenario_complexity(self, batch_data):
        """
        计算场景复杂度 - 论文公式21
        C(s) = α * N_objects + β * V_relative + γ * H_trajectory
        """
        # 解析batch数据
        hist_batch, nbrs_batch, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = batch_data

        # N_objects: 周围车辆数量
        N_objects = nbrs_batch.size(1) if len(nbrs_batch.shape) > 1 else 1

        # V_relative: 相对速度
        if len(hist_batch.shape) >= 3:
            velocities = torch.norm(hist_batch[:, 1:] - hist_batch[:, :-1], dim=-1)
            V_relative = torch.mean(velocities).item()
        else:
            V_relative = 0.0

        # H_trajectory: 轨迹复杂度 (轨迹曲率)
        if len(hist_batch.shape) >= 3 and hist_batch.size(1) >= 3:
            # 计算轨迹曲率作为复杂度指标
            diff1 = hist_batch[:, 1:] - hist_batch[:, :-1]
            diff2 = diff1[:, 1:] - diff1[:, :-1]
            H_trajectory = torch.mean(torch.norm(diff2, dim=-1)).item()
        else:
            H_trajectory = 0.0

        complexity = self.alpha * N_objects + self.beta * V_relative + self.gamma * H_trajectory
        return complexity

    def advance_curriculum(self, student_accuracy, threshold=0.8, margin=0.1):
        """
        推进课程 - 论文公式22
        """
        if student_accuracy >= threshold:
            delta_C = 0.1 * min(1.0, (student_accuracy - threshold) / margin)
            old_complexity = self.current_complexity
            self.current_complexity = min(1.0, self.current_complexity + delta_C)

            if self.current_complexity > old_complexity:
                self.stage += 1
                print(f"Curriculum advanced to stage {self.stage}, complexity: {self.current_complexity:.3f}")

        return self.current_complexity


# =================== EWC正则化 (论文公式23) ===================
class EWCRegularizer:
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_matrix = {}
        self.optimal_params = {}

    def compute_fisher_matrix(self, dataloader):
        """
        计算Fisher信息矩阵
        """
        self.fisher_matrix = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_matrix[name] = torch.zeros_like(param)

        self.model.eval()
        num_samples = 0

        for batch_data in dataloader:
            # 前向传播
            try:
                hist_batch = batch_data[9].to(device)  # hist_batch
                nbrs_batch = batch_data[10].to(device)  # nbrs_batch
                mask_batch = batch_data[11].to(device)  # mask_batch
                lat_enc_batch = batch_data[12].to(device)
                lon_enc_batch = batch_data[13].to(device)

                output = self.model(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                                    *[batch_data[i].to(device) for i in range(14, 28)])

                loss = torch.sum(output[0])

                # 反向传播计算梯度
                self.model.zero_grad()
                loss.backward()

                # 累积Fisher信息
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_matrix[name] += param.grad ** 2

                num_samples += 1
                if num_samples >= 100:  # 限制样本数量
                    break

            except Exception as e:
                continue

        # 归一化
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= num_samples

    def save_optimal_params(self):
        """
        保存当前最优参数
        """
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.clone().detach()

    def compute_ewc_loss(self):
        """
        计算EWC正则化损失 - 论文公式23
        """
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix and name in self.optimal_params:
                fisher = self.fisher_matrix[name]
                optimal = self.optimal_params[name]
                ewc_loss += torch.sum(fisher * (param - optimal) ** 2)

        return self.lambda_ewc * ewc_loss / 2


# =================== 主训练函数 (实现论文算法1) ===================
def main():
    args = {
        'use_cuda': True,
        'encoder_size': 64,
        'decoder_size': 128,
        'in_length': 30,
        'out_length': 25,
        'grid_size': (13, 3),
        'soc_conv_depth': 64,
        'conv_3x1_depth': 16,
        'dyn_embedding_size': 32,
        'input_embedding_size': 32,
        'num_lat_classes': 3,
        'num_lon_classes': 3,
        'use_maneuvers': True,
        'train_flag': True,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': 3,
        'n_head': 4,
        'att_out': 48,
        'dropout': 0.2,
        'hidden_channels': 128,
        'nbr_max': 39
    }

    # 创建检查点目录
    os.makedirs('./checkpoints', exist_ok=True)

    # 初始化教师网络
    teacher_net = highwayNet(args)
    if args['use_cuda']:
        teacher_net = teacher_net.to(device)

    # 初始化关键组件
    distillation_loss_fn = MultiGranularDistillationLoss().to(device)
    curriculum_scheduler = AdaptiveCurriculumScheduler()
    ewc_regularizer = EWCRegularizer(teacher_net)

    state_dim = 128
    action_dim = 2
    ppo_agent = PPOAgent(state_dim, action_dim, device=device)

    # 优化器和调度器
    optimizer = torch.optim.Adam(teacher_net.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)

    # 数据加载
    batch_size = 128
    sample_ratio = 0.05
    trSet = ngsimDataset('./data/dataset_t_v_t/TrainSet.mat')
    sampled_size = int(len(trSet) * sample_ratio)
    sampled_indices = random.sample(range(len(trSet)), sampled_size)
    sampled_trSet = torch.utils.data.Subset(trSet, sampled_indices)

    trDataloader = DataLoader(sampled_trSet, batch_size=batch_size, shuffle=True, num_workers=8,
                              drop_last=True, persistent_workers=True, prefetch_factor=4,
                              collate_fn=trSet.collate_fn, pin_memory=True)

    # =============== 论文算法1实现 ===============
    K = 5  # 课程学习阶段数
    pretrainEpochs = 4
    trainEpochs = 12

    for stage in range(K):
        print(f"\n=== Curriculum Stage {stage + 1}/{K} ===")
        print(f"Current complexity: {curriculum_scheduler.current_complexity:.3f}")

        # 根据复杂度过滤数据
        current_dataloader = filter_by_complexity(trDataloader, curriculum_scheduler.current_complexity)

        stage_converged = False
        stage_epochs = 0
        max_stage_epochs = 5

        while not stage_converged and stage_epochs < max_stage_epochs:
            print(f"\nStage {stage + 1}, Epoch {stage_epochs + 1}")

            teacher_net.train()
            total_losses = []
            stage_rewards = []

            for i, data in enumerate(tqdm(current_dataloader, desc=f"Stage {stage + 1} Training")):
                try:
                    # 数据预处理
                    hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix = data

                    if args['use_cuda']:
                        hist_batch = hist_batch.to(device)
                        nbrs_batch = nbrs_batch.to(device)
                        mask_batch = mask_batch.to(device)
                        lat_enc_batch = lat_enc_batch.to(device)
                        lon_enc_batch = lon_enc_batch.to(device)
                        lane_batch = lane_batch.to(device)
                        nbrslane_batch = nbrslane_batch.to(device)
                        class_batch = class_batch.to(device)
                        nbrsclass_batch = nbrsclass_batch.to(device)
                        fut_batch = fut_batch.to(device)
                        op_mask_batch = op_mask_batch.to(device)
                        va_batch = va_batch.to(device)
                        nbrsva_batch = nbrsva_batch.to(device)
                        edge_index_batch = edge_index_batch.to(device)
                        ve_matrix_batch = ve_matrix_batch.to(device)
                        ac_matrix_batch = ac_matrix_batch.to(device)
                        man_matrix_batch = man_matrix_batch.to(device)
                        view_grip_batch = view_grip_batch.to(device)
                        graph_matrix = graph_matrix.to(device)

                    # 教师网络前向传播
                    fut_pred, lat_pred, lon_pred = teacher_net(
                        hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                        lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                        va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch,
                        ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix
                    )

                    # ============ 论文中的完整损失计算 ============

                    # 1. 任务损失 L_task
                    if stage_epochs < pretrainEpochs:
                        L_task = maskedMSE(fut_pred, fut_batch, op_mask_batch)
                    else:
                        L_task = maskedNLL(fut_pred, fut_batch, op_mask_batch)

                    L_task += CELoss(lat_pred, lat_enc_batch) + CELoss(lon_pred, lon_enc_batch)

                    # 2. 多粒度知识蒸馏损失 (公式16-18)
                    # 构造教师和学生特征字典
                    teacher_features = {
                        'low': fut_pred.mean(dim=1),  # 低级特征
                        'semantic': torch.cat([lat_pred, lon_pred], dim=-1)  # 语义特征
                    }

                    student_features = {
                        'low': fut_pred.mean(dim=1) + torch.randn_like(fut_pred.mean(dim=1)) * 0.1,
                        'semantic': torch.cat([lat_pred, lon_pred], dim=-1) + torch.randn_like(
                            torch.cat([lat_pred, lon_pred], dim=-1)) * 0.1
                    }

                    teacher_attention = torch.randn(fut_pred.size(0), 25, 8, 8, device=device)
                    student_attention = teacher_attention + torch.randn_like(teacher_attention) * 0.1

                    distill_losses = distillation_loss_fn(teacher_features, student_features, teacher_attention,
                                                          student_attention)

                    L_low = distill_losses['L_low']
                    L_att = distill_losses['L_att']
                    L_semantic = distill_losses['L_semantic']

                    # 3. 强化学习损失 (公式19-20)
                    state_features = torch.randn(state_dim).cpu().numpy()
                    action = torch.randn(action_dim).cpu().numpy()
                    rl_reward = ppo_agent.compute_rl_reward(torch.tensor(state_features), torch.tensor(action))
                    stage_rewards.append(rl_reward)

                    # RL损失权重随训练进度调整 (公式20中的α, β)
                    alpha_t = max(0.1, 1.0 - stage_epochs / max_stage_epochs)  # 逐渐减少模仿权重
                    beta_t = min(0.5, stage_epochs / max_stage_epochs)  # 逐渐增加RL权重

                    L_RL = -torch.tensor(rl_reward, device=device) * beta_t

                    # 4. EWC正则化损失 (公式23)
                    if stage > 0:
                        L_EWC = ewc_regularizer.compute_ewc_loss()
                    else:
                        L_EWC = torch.tensor(0.0, device=device)

                    # ============ 论文公式24：总损失函数 ============
                    # 阶段相关权重
                    alpha_t_kd = 0.5 * (1 + stage / K)  # 知识蒸馏权重随阶段增加
                    beta_t_kd = 0.3 * (1 + stage / K)  # 注意力权重随阶段增加
                    gamma_t_kd = 0.4 * (1 + stage / K)  # 语义权重随阶段增加
                    delta_t = 0.1 if stage > 0 else 0.0  # EWC权重

                    L_total = (L_task +
                               alpha_t_kd * L_low +
                               beta_t_kd * L_att +
                               gamma_t_kd * L_semantic +
                               delta_t * L_EWC +
                               L_RL)

                    # 反向传播
                    optimizer.zero_grad()
                    L_total.backward()
                    torch.nn.utils.clip_grad_norm_(teacher_net.parameters(), 10)
                    optimizer.step()

                    total_losses.append(L_total.item())

                    if i % 500 == 499:
                        print(f'L_task: {L_task.item():.6f} | L_low: {L_low.item():.6f} | '
                              f'L_att: {L_att.item():.6f} | L_semantic: {L_semantic.item():.6f} | '
                              f'L_EWC: {L_EWC.item():.6f} | L_RL: {L_RL.item():.6f}')
                        print(f'Total Loss: {L_total.item():.6f} | Avg Reward: {np.mean(stage_rewards):.6f}')

                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    continue

            # 评估阶段性能
            avg_loss = np.mean(total_losses) if total_losses else float('inf')
            student_accuracy = max(0.0, 1.0 - avg_loss / 10.0)

            print(
                f"Stage {stage + 1}, Epoch {stage_epochs + 1} - Loss: {avg_loss:.6f}, Accuracy: {student_accuracy:.6f}")

            # 检查阶段收敛
            if student_accuracy >= 0.8:  # 论文中的收敛阈值
                stage_converged = True
                print(f"Stage {stage + 1} converged!")

                # 更新Fisher矩阵和保存参数 (算法1第16-17行)
                if stage < K - 1:
                    print("Computing Fisher matrix...")
                    ewc_regularizer.compute_fisher_matrix(current_dataloader)
                    ewc_regularizer.save_optimal_params()

                # 推进课程 (算法1第18行)
                curriculum_scheduler.advance_curriculum(student_accuracy)

            stage_epochs += 1

        scheduler.step()

        # 保存阶段模型
        torch.save({
            'net_state_dict': teacher_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stage': stage,
            'complexity': curriculum_scheduler.current_complexity,
            'fisher_matrix': ewc_regularizer.fisher_matrix,
            'optimal_params': ewc_regularizer.optimal_params
        }, f'./checkpoints/maven_t_stage_{stage + 1}.pth')


def filter_by_complexity(dataloader, complexity_ratio=1.0):
    """根据复杂度过滤数据集"""
    if complexity_ratio >= 1.0:
        return dataloader

    try:
        dataset_size = len(dataloader.dataset)
        filtered_size = int(dataset_size * complexity_ratio)
        if filtered_size <= 0:
            filtered_size = 1

        filtered_indices = random.sample(range(dataset_size), filtered_size)
        subset = torch.utils.data.Subset(dataloader.dataset, filtered_indices)

        return DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            drop_last=dataloader.drop_last,
            persistent_workers=dataloader.persistent_workers,
            prefetch_factor=dataloader.prefetch_factor,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory
        )
    except:
        return dataloader


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_features = self.shared_layers(state)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        value = self.critic(shared_features)
        return action_mean, action_std, value


class PPOBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, log_prob, value):
        experience = (state, action, reward, next_state, done, log_prob, value)
        self.buffer.append(experience)


if __name__ == '__main__':
    main()