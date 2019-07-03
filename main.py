import torch
import argparse
from config import config as cfg
import logging
import coloredlogs
import os
from torchvision import transforms as T

# Setup colorful logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


# noinspection PyShadowingNames
def train(model, optimizer, criterion, resume_from_epoch=0, max_val_accuracy=0):
    """
    Train the model
    :param model: Model to be trained
    :param optimizer: Method to compute gradients
    :param criterion: Criterion for computing loss
    :param resume_from_epoch: Resume training from this epoch
    :param max_val_accuracy: Save models with greater accuracy on validation set
    """
    model.train()
    train_loader = cfg.train_loader
    for epoch in range(resume_from_epoch, cfg.n_epochs):
        logger.info('TRAINING: Epoch {}/{}'.format(epoch+1, cfg.n_epochs))
        running_loss = 0
        step = 0
        for sample in train_loader:
            step += 1
            if step % cfg.print_freq == 0:
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

        if epoch % cfg.val_freq == 0:
            logger.info('VALIDATION')
            val_accuracy = val(model)
            logger.info(f'Validation accuracy: {val_accuracy}')
            if val_accuracy > max_val_accuracy:
                logger.info('Saving model...')
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': val_accuracy
                    }, cfg.model_path)
                except FileNotFoundError as fnf_error:
                    logger.error(f'{fnf_error}')
                else:
                    logger.info('Saved 🎉')
                max_val_accuracy = val_accuracy


# noinspection PyShadowingNames
def val(model):
    """
    Check model accuracy on validation set.
    :param model: Model to be tested
    :return: Validation accuracy
    """
    # TODO: Fix calculation of validation accuracy
    total_pixels = 0
    correct_pixels = 0
    val_accuracy = 0
    model.eval()
    val_loader = cfg.val_loader
    for sample in val_loader:
        sample['image'] = sample['image'].to(cfg.device)
        sample['label'] = sample['label'].type(torch.LongTensor).to(cfg.device)

        output = model(sample['image'])
        _, predicted = torch.max(output.data, 1)
        total_pixels += sample['label'].nelement()
        correct_pixels += predicted.eq(sample['label'].data).sum().item()
        val_accuracy = 100 * correct_pixels / total_pixels
    else:
        print('Validation Accuracy: {}'.format(val_accuracy))
        model.train()
        return val_accuracy


# noinspection PyShadowingNames
def test(model):
    """
    Get segmented image from trained model
    :param model: Model generating the mask
    :return: Segmented image
    """
    model.eval()
    test_loader = cfg.test_loader
    for sample in test_loader:
        sample['image'] = sample['image'].to(cfg.device)
        prediction = model(sample['image'])
        T.ToPILImage()(prediction).save(os.path.join(cfg.results_dir, 'output.jpeg'))
        print('Prediction saved in results/output.jpeg')


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description=f'CLI for {cfg.model_name}')
    parser.add_argument('--phase',
                        type=str,
                        default='train',
                        help='set phase[train(includes val)/test]')
    parser.add_argument('--load',
                        type=bool,
                        default=False,
                        help='load model from checkpoints/model.pth')
    args = parser.parse_args()

    # Load values from config file
    model = cfg.model
    optimizer = cfg.optimizer
    criterion = cfg.criterion
    resume_from_epoch = cfg.resume_from_epoch
    max_val_accuracy = cfg.max_val_accuracy

    if args.load:
        # Load values from checkpoint file
        checkpoint = torch.load(cfg.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_from_epoch = checkpoint['epoch']
        min_val_accuracy = checkpoint['val_accuracy']    # TODO: Add min_val_loss from checkpoint file

    if args.phase == 'train':
        train(model, optimizer, criterion, resume_from_epoch, max_val_accuracy)

    elif args.phase == 'test':
        test(model)

    else:
        raise ValueError('Choose one of train/test.')
