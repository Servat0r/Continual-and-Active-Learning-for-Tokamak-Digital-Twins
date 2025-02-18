from typing import Optional, TYPE_CHECKING, TextIO
import sys

from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ExperienceBalancedBuffer, ExemplarsBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

from ...buffers.pr import *


class PercentageReplayPlugin(ReplayPlugin):
    """
    Percentage Replay Plugin.

    Extends the ReplayPlugin to dynamically resize the buffer to represent
    a given percentage of the total training set after each experience.
    For example, if experiences sizes are {50000, 43000, 12000, 23000},
    and mem_percentage = 0.1, buffer sizes would be (at the beginning
    of each experience) {0, 5000, 9300, 10500} (hence total experience
    datasets would be {50000, 48000, 21300, 33500}, a net increase of
    {+0%, +11.63%, +77.5%, +45.65%}). It is also possible to specify
    a minimum size for the buffer.
    """
    def __init__(
        self,
        mem_percentage: float = 0.1,  # Percentage of total training set for the buffer
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        dump: bool = False, dump_fp: TextIO | str = sys.stdout,
        min_buffer_size: int = 0,
    ):
        super().__init__()
        assert 0 < mem_percentage <= 1, "Memory percentage must be between 0 and 1."
        self.mem_percentage = mem_percentage
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        self.total_training_examples = 0
        self.min_buffer_size = min_buffer_size

        self.storage_policy = storage_policy or ExperienceProportionalBuffer(
            max_size=self.min_buffer_size
        )
        self.dump = dump
        self.dump_fp = dump_fp if dump else None
        if isinstance(self.dump_fp, str):
            self.dump_fp = open(self.dump_fp, "w")
        self.storage_policy._dump_fp = self.dump_fp

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Update the memory buffer size dynamically based on the total
        training examples seen so far.
        """
        # Update the storage policy with new samples
        self.storage_policy.update(strategy, **kwargs)

        # Update the total training examples count using the current experience dataset
        current_exp_size = len(strategy.experience.dataset)
        self.total_training_examples += current_exp_size

        # Calculate new buffer size
        new_buffer_size = max(
            int(self.total_training_examples * self.mem_percentage),
            self.min_buffer_size
        )
        if self.dump_fp is not None:
            print(
                f"Buffer size after training experience {strategy.experience}:",
                new_buffer_size, sep=' ', end='\n', file=self.dump_fp, flush=True
            )

        print(f"Before resize: Buffer size = {len(self.storage_policy.buffer)}, "
              f"Groups = {len(self.storage_policy.buffer_groups)}")
        self.storage_policy.resize(strategy, new_buffer_size)
        print(f"After resize: Buffer size = {len(self.storage_policy.buffer)}, "
              f"Groups = {len(self.storage_policy.buffer_groups)}")

    def close(self):
        self.dump_fp.close()


__all__ = ["PercentageReplayPlugin"]
