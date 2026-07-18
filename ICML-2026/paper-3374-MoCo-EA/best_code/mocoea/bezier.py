"""Bezier curve optimization implementations."""

import torch
import torch.nn as nn
import torch.optim as optim
from mocoea.attacks import project_l1_ball

class BezierAdversarialUnconstrained:
    
    def __init__(self, model, norm='linf', eps=8/255, lr=0.01, num_iter=100,
                 normalize_fn=None):
        self.model = model
        self.norm = norm
        self.eps = eps
        self.lr = lr
        self.num_iter = num_iter
        self.normalize = normalize_fn or (lambda x: x)
    
    def bezier_curve(self, p0, p1, p2, t):
        return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    
    def project_norm_ball(self, delta):
        if self.norm == 'linf':
            return torch.clamp(delta, -self.eps, self.eps)
        elif self.norm == 'l2':
            delta_flat = delta.view(delta.size(0), -1)
            norm_delta = torch.norm(delta_flat, p=2, dim=1, keepdim=True) + 1e-10
            scale = torch.clamp(norm_delta / self.eps, min=1.0)
            delta_flat = delta_flat / scale
            return delta_flat.view_as(delta)
        elif self.norm == 'l1':
            return project_l1_ball(delta, self.eps)
    
    def optimize_setting_A(self, x, y, delta1, delta2, num_t_samples=20):
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            successful_points = 0
            
            t_values = torch.rand(num_t_samples).to(x.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                x_adv = torch.clamp(x + delta_t, 0, 1)
                x_adv_norm = self.normalize(x_adv)
                outputs = self.model(x_adv_norm)
                
                loss = -nn.CrossEntropyLoss()(outputs, y)
                total_loss += loss
                
                pred = outputs.argmax(dim=1)
                if pred != y:
                    successful_points += 1
            
            total_loss /= num_t_samples
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append(successful_points / num_t_samples)
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        print(f"Final theta norm: {theta_norms[-1]:.2f} × eps (unconstrained)")
        
        return theta.detach(), losses, success_rates, theta_norms
    
    def optimize_setting_B(self, x1, x2, y, delta1, delta2, num_t_samples=20):
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            success_x1 = 0
            success_x2 = 0
            success_both = 0
            
            t_values = torch.rand(num_t_samples).to(x1.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                x1_adv = torch.clamp(x1 + delta_t, 0, 1)
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                
                outputs1 = self.model(self.normalize(x1_adv))
                outputs2 = self.model(self.normalize(x2_adv))
                
                loss1 = -nn.CrossEntropyLoss()(outputs1, y)
                loss2 = -nn.CrossEntropyLoss()(outputs2, y)
                total_loss += (loss1 + loss2) / 2
                
                pred1 = outputs1.argmax(dim=1)
                pred2 = outputs2.argmax(dim=1)
                
                if pred1 != y:
                    success_x1 += 1
                if pred2 != y:
                    success_x2 += 1
                if pred1 != y and pred2 != y:
                    success_both += 1
            
            total_loss /= num_t_samples
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'x1': success_x1 / num_t_samples,
                'x2': success_x2 / num_t_samples,
                'both': success_both / num_t_samples
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        print(f"Final theta norm: {theta_norms[-1]:.2f} × eps (unconstrained)")
        return theta.detach(), losses, success_rates, theta_norms
    
    def optimize_setting_C(self, x1, x2, y1, y2, delta1, delta2, num_t_samples=20):
        theta = ((delta1 + delta2) / 2).clone().detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=self.lr)
        
        losses = []
        success_rates = []
        theta_norms = []
        
        for iteration in range(self.num_iter):
            optimizer.zero_grad()
            total_loss = 0
            success_x1 = 0
            success_x2 = 0
            success_both = 0
            
            t_values = torch.rand(num_t_samples).to(x1.device)
            
            for t in t_values:
                delta_t = self.bezier_curve(delta1, theta, delta2, t)
                delta_t = self.project_norm_ball(delta_t)
                
                x1_adv = torch.clamp(x1 + delta_t, 0, 1)
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                
                outputs1 = self.model(self.normalize(x1_adv))
                outputs2 = self.model(self.normalize(x2_adv))
                
                loss1 = -nn.CrossEntropyLoss()(outputs1, y1)
                loss2 = -nn.CrossEntropyLoss()(outputs2, y2)
                total_loss += (loss1 + loss2) / 2
                
                pred1 = outputs1.argmax(dim=1)
                pred2 = outputs2.argmax(dim=1)
                
                if pred1 != y1:
                    success_x1 += 1
                if pred2 != y2:
                    success_x2 += 1
                if pred1 != y1 and pred2 != y2:
                    success_both += 1
            
            total_loss /= num_t_samples
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            success_rates.append({
                'x1': success_x1 / num_t_samples,
                'x2': success_x2 / num_t_samples,
                'both': success_both / num_t_samples
            })
            theta_norms.append(torch.norm(theta.data.flatten()).item() / self.eps)
        
        print(f"Final theta norm: {theta_norms[-1]:.2f} × eps (unconstrained)")
        return theta.detach(), losses, success_rates, theta_norms
