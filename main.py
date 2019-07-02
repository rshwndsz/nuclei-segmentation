import torch
import argparse
import os
from torchvision import transforms as T
import config as cfg


def train(model):
    # TODO: Add ability to load from a checkpoint save
    model.to(cfg.device)
    model.train()
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
            sample['label'] = sample['label'].to(cfg.device)

            prediction = model(sample['image'])
            loss = criterion(prediction, sample['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Epoch done! Loss =', running_loss/len(train_loader))


def val(model):
    model.to(cfg.device)
    model.eval()
    criterion = cfg.criterion
    val_loader = cfg.val_loader

    running_loss = 0
    for sample in val_loader:
        sample['image'] = sample['image'].to(cfg.device)
        sample['label'] = sample['label'].to(cfg.device)

        prediction = model(sample['image'])
        loss = criterion(prediction, sample['label'])

        running_loss += loss.item()
    else:
        print('Validation Loss: {}'.format(running_loss/len(val_loader)))


def test(model):
    model.to(cfg.device)
    model.eval()

    test_loader = cfg.test_loader
    for sample in test_loader:
        sample['image'] = sample['image'].to(cfg.device)

        prediction = model(sample['image'])
        T.ToPILImage(prediction).save(os.path.join(cfg.results_dir, 'output.jpeg'))
        print('Prediction saved in results/output.jpeg')


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
