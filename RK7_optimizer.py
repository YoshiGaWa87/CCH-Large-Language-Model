import torch
import torch.optim as optim

class RK7Optimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        """初始化7階龍格－庫塔優化器。
        此實現基於 Fehlberg RKF7(8) 方法的7階解，共包含11個階段。

        Parameters:
            params: 模型參數 (來自 model.parameters())
            lr (float): 學習率 (相當於時間步長 h)
        """
        defaults = dict(lr=lr)
        super(RK7Optimizer, self).__init__(params, defaults)

        # Fehlberg RKF7(8) 的中間步驟係數 (a_ij)
        self.a = [
            [2/27],                                                                 # k2
            [1/36, 1/12],                                                           # k3
            [1/24, 0, 1/8],                                                         # k4
            [1/6, 0, -1/2, 2/3],                                                    # k5
            [-1/63, 0, 5/21, -20/63, 125/252],                                      # k6
            [8/315, 0, 13/105, -26/315, 25/126, 31/315],                             # k7
            [2/35, 0, 0, -12/35, 32/35, 0, 6/35],                                    # k8
            [0, 0, -9/14, 34/35, -29/14, 0, 0, 9/14],                                # k9
            [41/840, 0, 0, 0, 34/105, 9/35, 9/280, 9/280, 0],                        # k10
            [0, 0, 0, 0, 0, 0, 0, 41/840, 41/840, 0]                                 # k11
        ]

        # 最終7階更新的權重 (b_i)
        self.b = [0, 0, 0, 0, 34/105, 9/35, 9/280, 9/280, 0, 41/840, 41/840]


    def step(self, closure=None):
        """執行一步 RK7 優化。

        Parameters:
            closure (callable): 一個閉包，它重新評估模型，計算損失，並返回損失。
        """
        if closure is None:
            raise ValueError("RK7Optimizer 需要一個閉包函數。")

        # 保存所有參數的初始狀態
        initial_states = []
        for group in self.param_groups:
            group_states = {id(p): p.data.clone() for p in group['params']}
            initial_states.append(group_states)

        lr = self.param_groups[0]['lr']
        k = [[] for _ in range(11)] # 儲存 k1 到 k11

        # --- 階段 1: k1 ---
        loss = closure()
        for group_idx, group in enumerate(self.param_groups):
            k[0].append([-p.grad.clone() if p.grad is not None else None for p in group['params']])

        # --- 中間階段 (k2 到 k11) ---
        for i in range(10): # 計算 k2 到 k11
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is not None:
                        initial_p = initial_states[group_idx][id(p)]
                        update = torch.zeros_like(initial_p)
                        # 根據 a_ij 係數計算中間更新量
                        for j in range(i + 1):
                            if k[j][group_idx][p_idx] is not None:
                                update.add_(k[j][group_idx][p_idx], alpha=self.a[i][j])
                        p.data.copy_(initial_p).add_(update, alpha=lr)

            loss = closure()
            for group_idx, group in enumerate(self.param_groups):
                k[i+1].append([-p.grad.clone() if p.grad is not None else None for p in group['params']])

        # --- 最終參數更新 ---
        for group_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group['params']):
                if p.grad is not None:
                    initial_p = initial_states[group_idx][id(p)]
                    final_update = torch.zeros_like(initial_p)

                    # 根據 b_i 係數計算最終更新量
                    for i in range(11):
                       if self.b[i] != 0 and k[i][group_idx][p_idx] is not None:
                           final_update.add_(k[i][group_idx][p_idx], alpha=self.b[i])

                    # 恢復初始狀態並應用最終的加權更新
                    p.data.copy_(initial_p).add_(final_update, alpha=lr)

        return loss
