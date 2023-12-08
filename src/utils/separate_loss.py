import itertools
from typing import Any

import torch


class SeparateLoss:
    def __init__(self, mix_num, device):
        self.mix_num = mix_num
        self.permutations = [torch.tensor(perm, device=device) 
                             for perm in itertools.permutations(range(mix_num))]
    
    @staticmethod
    def gen_loss_fn(dim):
        dim = tuple(range(1, dim))
        return lambda x, y:torch.mean(torch.abs(x-y), dim=dim)

    def cal_seploss(self, clean_list:list, est_list:list, loss_weight_list:list):
        losses = []
        for perm in self.permutations:
            loss = 0
            for i in range(len(clean_list)):
                clean = clean_list[i]
                est = est_list[i]
                w = loss_weight_list[i]

                loss_fn = self.gen_loss_fn(clean.dim())
                clean_perm = clean.index_select(1, perm)
                loss += w * loss_fn(clean_perm, est)
            losses.append(loss)
        losses = torch.stack(losses, dim=-1)
        loss, _ = torch.min(losses, dim=-1)
        return torch.mean(loss)
    
    def __call__(self, clean_list:list, est_list:list, loss_weight_list:list) -> Any:
        return self.cal_seploss(clean_list, est_list, loss_weight_list)
    

if __name__ == "__main__":
    sep_loss = SeparateLoss(3,"cuda:0")
    clean = [torch.rand((4,3,12,34)).to("cuda:0"),
             torch.rand((4,3,431)).to("cuda:0")]
    est = [torch.rand((4,3,12,34)).to("cuda:0"),
             torch.rand((4,3,431)).to("cuda:0")]
    print(sep_loss.cal_seploss(clean, est, [lambda x, y: torch.mean(torch.abs(x-y), dim=(1,2,3)), lambda x, y: torch.mean(torch.abs(x-y), dim=(1,2))],[0.5,0.5]))