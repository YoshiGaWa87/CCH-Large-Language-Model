import torch
import torch.optim as optim

class RK6Optimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        """Initializes a 6th-order Runge-Kutta optimizer.
        This implementation uses an 8-stage RKF6(5) method for its 6th-order update.

        Parameters:
            params: model parameters (from model.parameters())
            lr (float): learning rate (equivalent to time step h)
        """
        defaults = dict(lr=lr)
        super(RK6Optimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs one RK6 optimization step.

        Parameters:
            closure (callable): A closure that reevaluates the model,
                                computes the loss, and returns the loss.
        """
        if closure is None:
            raise ValueError("RK6Optimizer requires a closure function.")

        # Save the initial state of all parameters
        initial_states = []
        for group in self.param_groups:
            group_states = {id(p): p.data.clone() for p in group['params']}
            initial_states.append(group_states)

        # Assuming a single learning rate for simplicity
        lr = self.param_groups[0]['lr']
        k = [[] for _ in range(8)] # To store k1 to k8

        # --- Stage 1: k1 ---
        loss = closure()
        for group_idx, group in enumerate(self.param_groups):
            k[0].append([-p.grad.clone() if p.grad is not None else None for p in group['params']])

        # --- Intermediate Stages (k2 to k8) ---
        # Coefficients for the intermediate steps
        a = [
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40],
            [16/135, 6656/12825, 28561/56430, -9/50, 2/55],
            [25/216, 1408/2565, 2197/4104, -1/5, 0] # k8 uses k1, k3, k4, k5
        ]

        # Order of k's to use for each stage calculation
        k_indices = [
            [0],                 # k2 depends on k1
            [0, 1],              # k3 depends on k1, k2
            [0, 1, 2],           # k4 depends on k1, k2, k3
            [0, 1, 2, 3],        # k5 depends on k1, k2, k3, k4
            [0, 1, 2, 3, 4],     # k6 depends on k1..k5
            [0, 2, 3, 4, 5],     # k7 depends on k1, k3, k4, k5, k6
            [0, 2, 3, 4, 6]      # k8 depends on k1, k3, k4, k5, k7
        ]

        for i in range(7): # Calculate k2 through k8
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is not None:
                        initial_p = initial_states[group_idx][id(p)]
                        update = torch.zeros_like(initial_p)
                        for j, k_idx in enumerate(k_indices[i]):
                            if k[k_idx][group_idx][p_idx] is not None:
                                update.add_(k[k_idx][group_idx][p_idx], alpha=a[i][j])
                        p.data.copy_(initial_p).add_(update, alpha=lr)

            loss = closure()
            for group_idx, group in enumerate(self.param_groups):
                k[i+1].append([-p.grad.clone() if p.grad is not None else None for p in group['params']])


        # --- Final Parameter Update ---
        # Weights for the 6th-order update
        b = [41/840, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280]
        # k-values to use in the final sum (k1, k4, k5, k6, k7, k8)
        final_k_indices = [0, 3, 4, 5, 6, 7]

        for group_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    initial_p = initial_states[group_idx][id(p)]
                    final_update = torch.zeros_like(initial_p)

                    for i, k_idx in enumerate(final_k_indices):
                       if k[k_idx][group_idx][p_idx] is not None:
                           # We can get the actual b-value from its original index k_idx
                           final_update.add_(k[k_idx][group_idx][p_idx], alpha=b[k_idx])

                    # Restore initial state and apply the final weighted update
                    p.data.copy_(initial_p).add_(final_update, alpha=lr)

        return loss
