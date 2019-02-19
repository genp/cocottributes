import argparse
import torchvision.models as models


def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='COCO Attributes Training')
    parser.add_argument('attributes', metavar='ATTR',
                        help='path to attributes dataset',
                        default="")
    parser.add_argument('train_ann', metavar='TRAIN',
                        help='path to training annotions',
                        default="instances_train2014.json")
    parser.add_argument('val_ann', metavar='VAL',
                        help='path to validation annotions',
                        default="instances_val2014.json")
    parser.add_argument('--dataset-root', default=".")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg19)')
    parser.add_argument('--n_attrs', default=204, type=int, metavar='N',
                        help='number of attribute labels (default: 204)')
    parser.add_argument('--optimizer-type', default="Adam",
                        type=str, help="The type of optimizer to use")
    parser.add_argument('--freeze_base', dest="freeze_base",
                        action="store_true", help="Flag indicating if we want to freeze base model")
    parser.add_argument('--loss', default="MultiLabelSoftMarginLoss", type=str,
                        help="Loss Criterion",
                        choices=('MultiLabelSoftMarginLoss',))
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 1)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--save-dir', default='.',
                        help="The location where the weights should be saved")
    args = parser.parse_args()
    return args
