import os
import json
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, args, log_dir='./logs'):
        self.args = args
        
        # Create root log directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create experiment directory
        self.log_path = os.path.join(log_dir, self.args.exp_name)
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
    
        # Save config as json
        with open(os.path.join(self.log_path, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f)
    
        # Tensorboard logger
        tblog_path = os.path.join(self.log_path, 'tensorboard')
        self.writer = SummaryWriter(tblog_path)

        # Text logger
        self.logfile = open(os.path.join(self.log_path, 'log.txt'), 'w')
        
        # Checkpoint directory
        self.save_dir = os.path.join(self.log_path, 'checkpoints')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def save_checkpoint(self, model, optimizer, epoch, save_name='latest'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(self.save_dir, f'{save_name}.pth'))

def load_checkpoint(save_path):
    checkpoint = torch.load(save_path)
    return checkpoint['epoch'], checkpoint['model_state_dict'], checkpoint['optimizer_state_dict']