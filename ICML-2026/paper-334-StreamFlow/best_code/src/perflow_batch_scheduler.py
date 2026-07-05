"""
PeRFlow批量调度器 - 正确实现批量去噪

基于PeRFlow的Time_Windows机制，实现向量化的批量处理，
保持算法完整性的同时获得性能提升。
"""

import torch
from .scheduler_perflow import PeRFlowScheduler


class PeRFlowBatchScheduler(PeRFlowScheduler):
    """
    PeRFlow的批量优化版本
    
    关键思想：
    1. 利用PeRFlow Time_Windows的并行性
    2. 向量化计算所有时间步的参数
    3. 保持算法完整性，避免质量损失
    """
    
    def get_window_alpha_batch(self, timepoints_batch):
        """
        批量版本的窗口参数计算
        
        Args:
            timepoints_batch: [batch_size] 的时间点tensor
            
        Returns:
            批量计算的窗口参数
        """
        batch_size = timepoints_batch.shape[0]
        device = timepoints_batch.device
        
        # 初始化批量结果
        t_win_start_batch = torch.zeros_like(timepoints_batch)
        t_win_end_batch = torch.zeros_like(timepoints_batch)
        
        # 向量化窗口查找
        for i, (window_start, window_end) in enumerate(zip(
            self.time_windows.window_starts, 
            self.time_windows.window_ends
        )):
            # 找到属于当前窗口的时间点
            # 使用robust的数值比较（考虑精度）
            mask = (timepoints_batch > (window_end + 0.1 * self.time_windows.precision))
            
            t_win_start_batch[mask] = window_start
            t_win_end_batch[mask] = window_end
        
        # 批量计算其他参数
        t_win_len_batch = t_win_end_batch - t_win_start_batch
        t_interval_batch = timepoints_batch - t_win_start_batch  # 注意：这是负值
        
        # 批量计算alpha参数 - 修复设备问题
        idx_start_batch = (t_win_start_batch * self.config.num_train_timesteps - 1).long()
        idx_start_batch = torch.clamp(idx_start_batch, 0, len(self.alphas_cumprod) - 1)
        
        idx_end_batch = (t_win_end_batch * self.config.num_train_timesteps - 1).long()
        idx_end_batch = torch.clamp(idx_end_batch, 0, len(self.alphas_cumprod) - 1)
        
        # 确保alphas_cumprod在正确设备上
        alphas_cumprod_device = self.alphas_cumprod.to(device)
        alphas_cumprod_start_batch = alphas_cumprod_device[idx_start_batch]
        alphas_cumprod_end_batch = alphas_cumprod_device[idx_end_batch]
        
        # 计算gamma_s_e
        alpha_cumprod_s_e_batch = alphas_cumprod_start_batch / alphas_cumprod_end_batch
        gamma_s_e_batch = alpha_cumprod_s_e_batch ** 0.5
        
        return (t_win_start_batch, t_win_end_batch, t_win_len_batch, 
                t_interval_batch, gamma_s_e_batch, 
                alphas_cumprod_start_batch, alphas_cumprod_end_batch)
    
    def step_batch(
        self,
        model_output_batch: torch.FloatTensor,
        timestep_batch: torch.Tensor,
        sample_batch: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        批量版本的PeRFlow step方法
        
        一次性处理多个时间步，保持PeRFlow算法完整性
        """
        batch_size = model_output_batch.shape[0]
        
        if self.config.prediction_type == "diff_eps":
            pred_epsilon_batch = model_output_batch
            t_c_batch = timestep_batch.float() / self.config.num_train_timesteps
            
            # 批量计算窗口参数
            (t_s_batch, t_e_batch, _, c_to_s_batch, gamma_s_e_batch, 
             alphas_start_batch, alphas_end_batch) = self.get_window_alpha_batch(t_c_batch)
            
            # 批量计算lambda_s和eta_s
            lambda_s_batch = 1 / gamma_s_e_batch
            eta_s_batch = -1 * (1 - gamma_s_e_batch**2)**0.5 / gamma_s_e_batch
            
            # 批量计算lambda_t和eta_t - 关键的PeRFlow公式！
            denominator_batch = lambda_s_batch * (t_c_batch - t_s_batch) + (t_e_batch - t_c_batch)
            
            lambda_t_batch = (lambda_s_batch * (t_e_batch - t_s_batch)) / denominator_batch
            eta_t_batch = (eta_s_batch * (t_e_batch - t_c_batch)) / denominator_batch
            
            # 批量计算pred_win_end - 扩展维度以匹配样本张量
            lambda_t_expanded = lambda_t_batch.view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
            eta_t_expanded = eta_t_batch.view(-1, 1, 1, 1)        # [batch, 1, 1, 1]
            
            pred_win_end_batch = lambda_t_expanded * sample_batch + eta_t_expanded * pred_epsilon_batch
            
            # 批量计算velocity
            time_diff_batch = (t_e_batch - (t_s_batch + c_to_s_batch)).view(-1, 1, 1, 1)
            pred_velocity_batch = (pred_win_end_batch - sample_batch) / time_diff_batch
            
        elif self.config.prediction_type == "ddim_eps":
            # 类似的批量实现对于ddim_eps
            pred_epsilon_batch = model_output_batch
            t_c_batch = timestep_batch.float() / self.config.num_train_timesteps
            
            (t_s_batch, t_e_batch, _, c_to_s_batch, _, 
             alphas_start_batch, alphas_end_batch) = self.get_window_alpha_batch(t_c_batch)
            
            lambda_s_batch = (alphas_end_batch / alphas_start_batch)**0.5
            eta_s_batch = (1-alphas_end_batch)**0.5 - (alphas_end_batch / alphas_start_batch * (1-alphas_start_batch))**0.5
            
            denominator_batch = lambda_s_batch * (t_c_batch - t_s_batch) + (t_e_batch - t_c_batch)
            lambda_t_batch = (lambda_s_batch * (t_e_batch - t_s_batch)) / denominator_batch
            eta_t_batch = (eta_s_batch * (t_e_batch - t_c_batch)) / denominator_batch
            
            lambda_t_expanded = lambda_t_batch.view(-1, 1, 1, 1)
            eta_t_expanded = eta_t_batch.view(-1, 1, 1, 1)
            
            pred_win_end_batch = lambda_t_expanded * sample_batch + eta_t_expanded * pred_epsilon_batch
            time_diff_batch = (t_e_batch - (t_s_batch + c_to_s_batch)).view(-1, 1, 1, 1)
            pred_velocity_batch = (pred_win_end_batch - sample_batch) / time_diff_batch
            
        else:
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")
        
        # 批量计算dt和prev_sample
        # 找到下一个时间步 - 修复设备问题
        prev_timestep_batch = torch.zeros_like(timestep_batch)
        timesteps_device = self.timesteps.to(device)
        for i, t in enumerate(timestep_batch):
            idx = torch.where(timesteps_device == t)[0]
            if len(idx) > 0 and idx[0] + 1 < len(timesteps_device):
                prev_timestep_batch[i] = timesteps_device[idx[0] + 1]
            else:
                prev_timestep_batch[i] = 0
        
        dt_batch = (prev_timestep_batch - timestep_batch).float() / self.config.num_train_timesteps
        dt_expanded = dt_batch.view(-1, 1, 1, 1)
        
        # 批量更新样本
        prev_sample_batch = sample_batch + dt_expanded * pred_velocity_batch
        
        if not return_dict:
            return (prev_sample_batch,)
        
        return [prev_sample_batch]  # 简化的返回格式
        
    def scheduler_step_batch_perflow(
        self, 
        model_pred_batch: torch.Tensor, 
        x_t_latent_batch: torch.Tensor, 
        timestep_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        为StreamDiffusion兼容的批量step接口
        
        这是核心的批量去噪方法，真正实现4步并行处理！
        """
        # 使用我们的批量step方法
        result = self.step_batch(
            model_pred_batch, 
            timestep_batch, 
            x_t_latent_batch, 
            return_dict=False
        )
        
        return result[0]  # 返回prev_sample


def create_perflow_batch_scheduler(original_scheduler: PeRFlowScheduler) -> PeRFlowBatchScheduler:
    """
    将现有的PeRFlowScheduler转换为批量版本
    """
    # 创建批量调度器
    batch_scheduler = PeRFlowBatchScheduler.from_config(original_scheduler.config)
    
    # 复制状态
    batch_scheduler.alphas_cumprod = original_scheduler.alphas_cumprod
    batch_scheduler.timesteps = original_scheduler.timesteps
    batch_scheduler.time_windows = original_scheduler.time_windows
    
    return batch_scheduler