from typing import Dict
import numpy as np
import qpsolvers
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class ConstantSizeGEMPlugin(SupervisedPlugin):

    def __init__(self, mem_size: int, memory_strength: float):

        super().__init__()

        self.mem_size = mem_size
        self.memory_strength = memory_strength

        self.memory_x: Dict[int, Tensor] = dict()
        self.memory_y: Dict[int, Tensor] = dict()
        self.memory_tid: Dict[int, Tensor] = dict()

        self.G: Tensor = torch.empty(0)
    
    def get_memory_sizes(self, current_exp: int) -> list[int]:
        base_value = self.mem_size // (current_exp + 1)
        results = [base_value for _ in range(current_exp + 1)]
        residuals = self.mem_size - base_value * (current_exp + 1)
        for i in range(residuals):
            results[i % (current_exp + 1)] += 1
        return results

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if strategy.experience.current_experience > 0:
            G = []
            strategy.model.train()
            for t in range(strategy.clock.train_exp_counter):
                strategy.model.train()
                strategy.optimizer.zero_grad()
                xref = self.memory_x[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
                loss = strategy._criterion(out, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            (
                                p.grad.flatten()
                                if p.grad is not None
                                else torch.zeros(p.numel(), device=strategy.device)
                            )
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if strategy.experience.current_experience > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.update_memory(
            strategy.experience.dataset,
            strategy.experience.current_experience,
            strategy.train_mb_size,
        )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current experience.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        memory_sizes = self.get_memory_sizes(t)
        # Resize previous experiences
        for i in range(t):
            current_size = len(self.memory_x[i])
            target_size = memory_sizes[i]
            if current_size > target_size:
                self.memory_x[i] = self.memory_x[i][:target_size]
                self.memory_y[i] = self.memory_y[i][:target_size]
                self.memory_tid[i] = self.memory_tid[i][:target_size]
        current_exp_target_size = memory_sizes[t]
        tot = 0
        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            if tot + x.size(0) <= current_exp_target_size:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

            else:
                diff = current_exp_target_size - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat(
                        (self.memory_tid[t], tid[:diff]), dim=0
                    )
                break
            tot += x.size(0)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        # solution with old quadprog library, same as the author's implementation
        # v = quadprog.solve_qp(P, q, G, h)[0]
        # using new library qpsolvers
        v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()


__all__ = ['ConstantSizeGEMPlugin']