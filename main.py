import torch
import argparse
from config import config as cfg
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def train(model):
    # TODO: Add ability to load from a checkpoint save
    # See: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    model.to(cfg.device)
    optimizer = cfg.optimizer
    criterion = cfg.criterion
    train_loader = cfg.train_loader

    for e in range(cfg.n_epochs):
        logger.info('Epoch {}/{}'.format(e+1, cfg.n_epochs))
        running_loss = 0
        step = 0
        for sample in train_loader:
            step += 1
            logger.info(f'Image {step}/{len(train_loader)}')
            sample['image'] = sample['image'].to(cfg.device)
            sample['label'] = sample['label'].type(torch.LongTensor).to(cfg.device)

            loss = criterion(model(sample['image']), sample['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            logger.info(f'Epoch done! Loss = {running_loss/len(train_loader)}')
            logger.info('Saving model...')
            try:
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                }, cfg.model_path)
            except FileNotFoundError as fnf_error:
                logger.error(f'Unable to save.: {fnf_error}')
            else:
                logger.info('Saved 🎉')


def val():
    # TODO: Implement this from 'refactor'
    pass


def test():
    # TODO: Implement this from 'refactor'
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'CLI for {cfg.model_name}')
    parser.add_argument('--phase', type=str, default='train')
    args = parser.parse_args()

    if args.phase == 'train':
        train(cfg.model)

    elif args.phase == 'test':
        test()

    elif args.phase == 'validate':
        val()

    else:
        raise ValueError('Choose one of train/validate/test.')
