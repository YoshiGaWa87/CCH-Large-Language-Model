import torch
import torch.optim as optim

class RK8Optimizer(optim.Optimizer):
    def __init__(self, params, lr=1e-2):
        """初始化8階龍格－庫塔優化器。
        此實現基於 Dormand-Prince DOPRI8(7) 方法，共包含12個階段。

        Parameters:
            params: 模型參數 (來自 model.parameters())
            lr (float): 學習率 (相當於時間步長 h)
        """
        defaults = dict(lr=lr)
        super(RK8Optimizer, self).__init__(params, defaults)

        # Dormand-Prince 8(7) 的中間步驟係數 (a_ij)
        # 為了簡潔，我們將在 step 函數中直接使用它們

    def step(self, closure=None):
        """執行一步 RK8 優化。"""
        if closure is None:
            raise ValueError("RK8Optimizer 需要一個閉包函數。")

        # 保存所有參數的初始狀態
        initial_states = []
        for group in self.param_groups:
            group_states = {id(p): p.data.clone() for p in group['params']}
            initial_states.append(group_states)

        lr = self.param_groups[0]['lr']
        k = [[] for _ in range(12)] # 儲存 k1 到 k12

        # 係數 a_ij
        a = [
            [1/18],
            [1/24, 1/8],
            [5/12, -25/12, 25/12],
            [1/20, 0, 0, 1/5],
            [-25/108, 0, 0, 125/108, -65/27],
            [125/33, 0, 0, -512/33, 392/11, -21/11],
            [55/2, 0, 0, -2560/11, 1536/11, -42/11, -35/2],
            [-10549/1260, 0, 0, 41492/297, -32528/297, 2197/330, 28/5, 0],
            [2682/35, 0, 0, -100916/105, 81088/105, -1179/11, -2197/35, 334/35, 0],
            [-205/28, 0, 0, 0, 0, 125/28, 0, 0, 0, 0],
            [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/280, 9/280, 0, 0] # k12的係數與RKF7(8)的8階解相同
        ]

        # 最終8階更新的權重 (b_i)
        b = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/280, 9/280, 0, 41/840, 0]


        # --- 階段 1: k1 ---
        loss = closure()
        for group_idx, group in enumerate(self.param_groups):
            k[0].append([-p.grad.clone() if p.grad is not None else None for p in group['params']])

        # --- 中間階段 (k2 到 k12) ---
        for i in range(11): # 計算 k2 到 k12
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is not None:
                        initial_p = initial_states[group_idx][id(p)]
                        update = torch.zeros_like(initial_p)
                        # 根據 a_ij 係數計算中間更新量
                        for j in range(i + 1):
                            if j < len(a[i]) and a[i][j] != 0 and k[j][group_idx][p_idx] is not None:
                                update.add_(k[j][group_idx][p_idx], alpha=a[i][j])
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
                    for i in range(12):
                       if b[i] != 0 and k[i][group_idx][p_idx] is not None:
                           final_update.add_(k[i][group_idx][p_idx], alpha=b[i])

                    p.data.copy_(initial_p).add_(final_update, alpha=lr)

        return loss
