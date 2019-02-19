"""
python main_convnet.py cocottributes_eccv_version_py3.jbl instances_train2014.json instances_val2014.json
"""


import arguments
import os
import logging

import torch
from torch import optim
import torch.utils.data as data
import torchvision.transforms as transforms

import trainer
from dataset import COCOAttributes
from model_factory import get_network, get_optimizer, get_criterion
import evaluation

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("attributes")


def main():
    args = arguments.parse()

    checkpoint = args.checkpoint if args.checkpoint else None

    model, params = get_network(args.arch, args.n_attrs,
                                checkpoint=checkpoint, base_frozen=args.freeze_base)

    criterion = get_criterion(loss_type=args.loss, args=args)

    optimizer = get_optimizer(params,
                              fc_lr=float(args.lr),
                              opt_type=args.optimizer_type,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.1,
                                          last_epoch=args.start_epoch - 1)
    if checkpoint:
        state = torch.load(checkpoint)
        model.load_state_dict(state["state_dict"])
        scheduler.load_state_dict(state['scheduler'])

    # Dataloader code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    logger.info("Setting up training data")
    train_loader = data.DataLoader(
        COCOAttributes(args.attributes, args.train_ann, train=True, split='train2014',
                       transforms=train_transforms,
                       dataset_root=args.dataset_root),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    logger.info("Setting up validation data")
    val_loader = data.DataLoader(
        COCOAttributes(args.attributes, args.val_ann, train=False, split='val2014',
                       transforms=val_transforms,
                       dataset_root=args.dataset_root),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    best_prec1 = 0

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info("Beginning training...")

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        trainer.train(train_loader, model, criterion,
                      optimizer, epoch, args.print_freq)

        # evaluate on validation set
        # trainer.validate(val_loader, model, criterion, epoch, args.print_freq)
        prec1 = 0

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)
        trainer.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'loss': args.loss,
            'optimizer': args.optimizer_type,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'batch_size': args.batch_size,
            'best_prec1': best_prec1,
        }, args.save_dir, '{0}_{1}_checkpoint.pth.tar'.format(args.arch, args.loss).lower())

    logger.info('Finished Training')

    logger.info('Running evaluation')
    evaluator = evaluation.Evaluator(model, val_loader,
                                     batch_size=args.batch_size,
                                     name="{0}_{1}".format(args.arch, args.loss))
    with torch.no_grad():
        evaluator.evaluate()


if __name__ == "__main__":
    main()
