import os
from avalanche.logging import CSVLogger


class CustomCSVLogger(CSVLogger):
    def __init__(
            self, log_folder: str = None, train_results_filename: str = 'train_results.csv',
            eval_results_filename: str = 'eval_results.csv'
    ):
        super(CSVLogger, self).__init__()
        self.log_folder = log_folder if log_folder is not None else "csvlogs"
        os.makedirs(self.log_folder, exist_ok=True)

        self.training_file = open(
            os.path.join(self.log_folder, train_results_filename), "w"
        )
        self.eval_file = open(
            os.path.join(self.log_folder, eval_results_filename), "w"
        )
        os.makedirs(self.log_folder, exist_ok=True)

        # current training experience id
        self.training_exp_id = None

        # if we are currently training or evaluating
        # evaluation within training will not change this flag
        self.in_train_phase = None

        # validation metrics computed during training
        self.val_acc, self.val_loss = 0, 0

        # print csv headers
        print(
            "training_exp",
            "epoch",
            "training_accuracy",
            "val_accuracy",
            "training_loss",
            "val_loss",
            sep=",",
            file=self.training_file,
            flush=True,
        )
        print(
            "eval_exp",
            "training_exp",
            "eval_accuracy",
            "eval_loss",
            "forgetting",
            sep=",",
            file=self.eval_file,
            flush=True,
        )


__all__ = ["CustomCSVLogger"]
