from typing import TYPE_CHECKING

from avalanche.training import BalancedExemplarsBuffer, ReservoirSamplingBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate


from ..misc import stdout_debug_print


class ExperienceProportionalBuffer(BalancedExemplarsBuffer[ReservoirSamplingBuffer]):

    def __init__(self, max_size: int, dump_fp=None):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        """
        super().__init__(max_size, adaptive_size=True)
        self._num_exps = 0
        self._exps_lengths = []
        self._total_exps_length = 0
        self._dump_fp = dump_fp
    
    def add_exp(self, exp_length):
        self._exps_lengths.append(exp_length)
        self._total_exps_length += exp_length

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if num_groups != len(self._exps_lengths):
            raise RuntimeError(
                f"Invalid buffer state for getting lengths for {num_groups} groups: " + \
                "only {len(self._exps_lengths)} available!"
            )
        factor = self.max_size / self._total_exps_length
        lengths = [int(factor * self._exps_lengths[i]) for i in range(num_groups)]
        residual_lengths = self.max_size - sum(lengths)
        for j in range(residual_lengths):
            lengths[j % len(lengths)] += 1
        log_str = f"Buffer Lengths for {num_groups} groups = {lengths}"
        # We don't need excess lengths, as int(x) <= x for all x in R
        stdout_debug_print(log_str, color='green')
        if self._dump_fp is not None:
            print(log_str, file=self._dump_fp, flush=True)
        return lengths

    def post_adapt(self, agent, exp):
        self._num_exps += 1
        new_data = exp.dataset
        self.add_exp(len(new_data))
        lens = self.get_group_lengths(self._num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(agent, ll)


__all__ = ['ExperienceProportionalBuffer']
