import torch
import gc


def get_resident_tensor(logger):
    objs = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                logger.info(f'{type(obj)}, {obj.size()}')
                objs += 1
        except:
            pass
    logger.info(f'Number of resident tensors: {objs}')
