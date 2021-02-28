from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def log_loss(self, tag, loss, iteration):
        self.writer.add_scalar(tag, loss, iteration)

    def save_images(self):
        x=2

    def close(self):
        self.writer.close()