import torch
import os
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, loss, path):
Path(path).parent.mkdir(parents=True, exist_ok=True)
state = {
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'epoch': epoch,
'loss': loss
}
torch.save(state, path)




def load_checkpoint(path, model, optimizer=None, map_location=None):
state = torch.load(path, map_location=map_location)
model.load_state_dict(state['model_state_dict'])
if optimizer and 'optimizer_state_dict' in state:
optimizer.load_state_dict(state['optimizer_state_dict'])
return state




def count_parameters(model):
return sum(p.numel() for p in model.parameters() if p.requires_grad)
