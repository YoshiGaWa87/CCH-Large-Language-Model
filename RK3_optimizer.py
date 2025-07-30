import torch
import torch.optim as optim

class RK4Optimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        """Initialize RK4 optimizer
        Parameters:
            params: model parameters (from model.parameters())
            lr (float): learning rate (equivalent to time step)
        """
        defaults = dict(lr=lr)
        super(RK4Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Perform one RK4 optimization step
        Parameters:
            closure (callable, optional): computes loss and gradients, must return loss and trigger backward pass
        """
        if closure is None:
            raise ValueError("RK4Optimizer requires a closure function that computes the loss and gradients.")

        # Save current parameter states for all parameters, using indices
        state_dicts = []
        for group in self.param_groups:
            state_dict = {idx: p.data.clone() for idx, p in enumerate(group['params'])}  # Include all params
            state_dicts.append(state_dict)

        # Perform RK4's four stages calculation
        k1, k2, k3, k4 = [], [], [], []
        lr = self.param_groups[0]['lr']  # Assuming a single learning rate for simplicity

        # k1: current gradient
        loss = closure()
        for group_idx, group in enumerate(self.param_groups):
            k1_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k1_group.append(-p.grad.clone())
                else:
                    k1_group.append(None)
            k1.append(k1_group)

        # k2: intermediate gradient (theta + lr/2 * k1)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k1[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_(lr / 2 * k1[group_idx][p_idx])
        loss = closure()
        for group in self.param_groups:
            k2_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k2_group.append(-p.grad.clone())
                else:
                    k2_group.append(None)
            k2.append(k2_group)

        # k3: intermediate gradient (theta + lr/2 * k2)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k2[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_(lr / 2 * k2[group_idx][p_idx])
        loss = closure()
        for group in self.param_groups:
            k3_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k3_group.append(-p.grad.clone())
                else:
                    k3_group.append(None)
            k3.append(k3_group)

        # k4: intermediate gradient (theta + lr * k3)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k3[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_(lr * k3[group_idx][p_idx])
        loss = closure()
        for group in self.param_groups:
            k4_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k4_group.append(-p.grad.clone())
                else:
                    k4_group.append(None)
            k4.append(k4_group)

        # Update parameters: theta = theta + lr/6 * (k1 + 2*k2 + 2*k3 + k4)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            k1_group = k1[group_idx]
            k2_group = k2[group_idx]
            k3_group = k3[group_idx]
            k4_group = k4[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k1_group[p_idx] is not None:
                    p.data.copy_(state_dict[p_idx])  # Restore initial state before final update
                    update = (k1_group[p_idx] + 2 * k2_group[p_idx] + 2 * k3_group[p_idx] + k4_group[p_idx]) * (lr / 6)
                    p.data.add_(update)

        return loss
