import torch
import argparse
import config as cfg
import torch.nn.functional as F


def train(model):
    # TODO: Add ability to load from a checkpoint save
    model.to(cfg.device)
    optimizer = cfg.optimizer
    criterion = cfg.criterion
    train_loader = cfg.train_loader

    for e in range(cfg.n_epochs):
        print('Epoch {}/{}'.format(e+1, cfg.n_epochs))
        running_loss = 0
        for sample in train_loader:
            print('Mem Used:', torch.cuda.memory_allocated(0))
            print('Mem Cached:', torch.cuda.memory_cached(0))

            sample['image'] = sample['image'].to(cfg.device)
            sample['label'] = sample['label'].type(torch.FloatTensor).to(cfg.device)
            print('sample label dtype:', sample['label'].type())

            prediction = model(sample['image'])
            loss = criterion(prediction, F.interpolate(sample['label'], prediction.size()[2:], mode='bilinear'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Epoch done! Loss =', running_loss/len(train_loader))


def val():
    pass


def test():
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
        raise ValueError('Choose one of train/validate/test')
