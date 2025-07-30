import torch
import torch.optim as optim

class RK5Optimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        """Initializing RK5 optimizer
        Parameters:
            params: model parameters (from model.parameters())
            lr (float): learning rate (equivalent to time step)
        """
        defaults = dict(lr=lr)
        super(RK5Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Perform one RK5 optimization step
        Parameters:
            closure (callable, optional): computes loss and gradients, must return loss and trigger backward pass
        """
        if closure is None:
            raise ValueError("RK5Optimizer requires a closure function that computes the loss and gradients.")

        # Save current parameter states for all parameters, using indices
        state_dicts = []
        for group in self.param_groups:
            state_dict = {idx: p.data.clone() for idx, p in enumerate(group['params'])}  # Include all params
            state_dicts.append(state_dict)

        # Perform RK5's six stages calculation
        k1, k2, k3, k4, k5, k6 = [], [], [], [], [], []
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

        # k2: intermediate gradient (x_i + 1/4 h, y_i + 1/4 k1 h)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k1[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_(lr / 4 * k1[group_idx][p_idx])
        loss = closure()
        for group in self.param_groups:
            k2_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k2_group.append(-p.grad.clone())
                else:
                    k2_group.append(None)
            k2.append(k2_group)

        # k3: intermediate gradient (x_i + 1/4 h, y_i + 1/8 k1 h + 1/8 k2 h)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k2[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_((lr / 8 * k1[group_idx][p_idx] + lr / 8 * k2[group_idx][p_idx]))
        loss = closure()
        for group in self.param_groups:
            k3_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k3_group.append(-p.grad.clone())
                else:
                    k3_group.append(None)
            k3.append(k3_group)

        # k4: intermediate gradient (x_i + 1/2 h, y_i - 1/2 k2 h + k3 h)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k3[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_((lr / 2 * k3[group_idx][p_idx] - lr / 2 * k2[group_idx][p_idx]))
        loss = closure()
        for group in self.param_groups:
            k4_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k4_group.append(-p.grad.clone())
                else:
                    k4_group.append(None)
            k4.append(k4_group)

        # k5: intermediate gradient (x_i + 3/4 h, y_i + 3/16 k1 h + 9/16 k4 h)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k4[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_((3 * lr / 16 * k1[group_idx][p_idx] + 9 * lr / 16 * k4[group_idx][p_idx]))
        loss = closure()
        for group in self.param_groups:
            k5_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k5_group.append(-p.grad.clone())
                else:
                    k5_group.append(None)
            k5.append(k5_group)

        # k6: intermediate gradient (x_i + h, y_i - 3/7 k1 h + 2/7 k2 h + 12/7 k3 h - 12/7 k4 h + 8/7 k5 h)
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k5[group_idx][p_idx] is not None:
                    p.data.copy_(state_dict[p_idx]).add_((-3 * lr / 7 * k1[group_idx][p_idx] + 2 * lr / 7 * k2[group_idx][p_idx] +
                                                        12 * lr / 7 * k3[group_idx][p_idx] - 12 * lr / 7 * k4[group_idx][p_idx] +
                                                        8 * lr / 7 * k5[group_idx][p_idx]))
        loss = closure()
        for group in self.param_groups:
            k6_group = []
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    k6_group.append(-p.grad.clone())
                else:
                    k6_group.append(None)
            k6.append(k6_group)

        # Update parameters: y_{i+1} = y_i + 1/90 (7k1 + 32k3 + 12k4 + 32k5 + 7k6) h
        for group_idx, group in enumerate(self.param_groups):
            state_dict = state_dicts[group_idx]
            k1_group = k1[group_idx]
            k3_group = k3[group_idx]
            k4_group = k4[group_idx]
            k5_group = k5[group_idx]
            k6_group = k6[group_idx]
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None and k1_group[p_idx] is not None:
                    p.data.copy_(state_dict[p_idx])  # Restore initial state before final update
                    update = (7 * k1_group[p_idx] + 32 * k3_group[p_idx] + 12 * k4_group[p_idx] +
                              32 * k5_group[p_idx] + 7 * k6_group[p_idx]) * (lr / 90)
                    p.data.add_(update)

        return loss
