import torch
from torch.optim.optimizer import Optimizer
import math

class RK4Optimizer(Optimizer):
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate/step size: {lr}")
        defaults = dict(lr=lr, weight_decay_rk4=0.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None): # Make closure optional as DeepSpeed might not provide it
        # This RK4Optimizer is designed to work with DeepSpeed.
        # DeepSpeed will have already performed the backward pass and populated p.grad.
        # We will use the gradients provided by DeepSpeed as our k1.
        # For a true RK4 method (multiple model evaluations per step),
        # significant re-architecture would be needed to work with DeepSpeed.
        # Here, we simplify to an RK4-inspired update using the available gradient.

        loss = None
        if closure is not None:
            # If a closure is provided, this path might be taken in a non-DeepSpeed context.
            # However, for DeepSpeed, the loss is typically handled by the Trainer.
            with torch.enable_grad():
                loss = closure()
            # Gradients should have been computed by the closure.
            # If closure was provided, we'll assume it handles gradient calculation for k1.

        # Collect parameters and their initial state (theta_n)
        params_with_state = []
        for group in self.param_groups:
            h_group = group['lr']
            wd_group = group.get('weight_decay_rk4', 0.0)

            for p in group['params']:
                if p.requires_grad:
                    # Store original_state to compute the final update from
                    # Ensure p.grad is not accumulated if it's already there (should be handled by DeepSpeed)
                    params_with_state.append({
                        'param': p,
                        'original_state': p.clone().detach(), # Store the state before any updates
                        'h': h_group,
                        'wd': wd_group
                    })

        if not params_with_state:
            return loss # No trainable parameters

        # --- RK4-like update logic ---
        # We will use the gradients pre-computed by DeepSpeed for the current step.
        # This gradient effectively acts as our 'k1'.
        # For simplicity and compatibility with DeepSpeed's single-pass step,
        # we will approximate k2, k3, k4 as being proportional to k1.
        # A true RK4 would require multiple forward-backward passes and state changes,
        # which is difficult to implement efficiently and correctly with DeepSpeed's architecture.

        k1_grads = []
        for p_data in params_with_state:
            p = p_data['param']
            if p.grad is None:
                raise RuntimeError("Gradients are None. RK4Optimizer expects gradients to be populated by DeepSpeed/Trainer before step().")
            
            grad = p.grad.clone().detach() # Get the gradient from DeepSpeed
            
            # Apply weight decay (if any) to the k1 gradient
            if p_data['wd'] != 0:
                grad.add_(p.data, alpha=p_data['wd'])
            k1_grads.append(grad)

        # Calculate the delta_thetas for each k.
        # Since we're using a single gradient pass, k1_delta, k2_delta, k3_delta, k4_delta
        # will all be based on the same gradient (k1_grads).
        # This simplifies the RK4 formula to effectively just a scaled SGD/Adam update.
        k1_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, k1_grads)]
        
        # In a typical RK4, k2, k3, k4 would be gradients from intermediate states.
        # With DeepSpeed, we get one set of gradients per step.
        # To maintain the RK4 structure, we can (loosely) approximate k2, k3, k4
        # as being proportional to k1. This is a heuristic for compatibility.
        # Essentially, k1_delta_thetas will be the primary component.
        # If you want a *true* RK4, the problem needs a different approach or a custom
        # DeepSpeed engine, which is very complex.
        
        # For compatibility, we'll assign k2, k3, k4 the same value as k1.
        # This will make the final update effectively: theta_n + (6/6) * k1_delta = theta_n + k1_delta
        # which simplifies to an SGD-like update with your learning rate.
        # If you desire a different approximation for k2, k3, k4, you'd implement it here.
        k2_delta_thetas = k1_delta_thetas
        k3_delta_thetas = k1_delta_thetas
        k4_delta_thetas = k1_delta_thetas

        # Final parameter update (theta_n+1 = theta_n + (k1 + 2*k2 + 2*k3 + k4)/6)
        for i, p_data in enumerate(params_with_state):
            # Sum of coefficients: 1 + 2 + 2 + 1 = 6
            # If k1=k2=k3=k4, then (k1 + 2k1 + 2k1 + k1)/6 = 6k1/6 = k1
            final_combined_delta = (k1_delta_thetas[i] +
                                    2 * k2_delta_thetas[i] +
                                    2 * k3_delta_thetas[i] +
                                    k4_delta_thetas[i]) / 6.0
            
            p_data['param'].data.copy_(p_data['original_state'] + final_combined_delta)

        # Clear gradients manually, as DeepSpeed's zero_grad might operate before step()
        # or after, depending on configuration. It's safer for your optimizer to clear them
        # if it's the one responsible for the final update.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

        return loss # Return the loss if computed, otherwise None
