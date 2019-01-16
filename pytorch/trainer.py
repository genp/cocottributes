import time
import torch
import os.path as osp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, save_dir, filename):
    torch.save(state, osp.join(save_dir, filename))


def adjust_learning_rate(optimizer, epoch, reduce_epoch=5):
    """Sets the learning rate to the initial LR decayed by 10 every reduce_epoch epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        # Decrease the current learning rate
        lr *= 0.1 if epoch > 0 and epoch % reduce_epoch == 0 else 1
        param_group['lr'] = lr
        # print(optimizer.param_groups)


def accuracy(output, target):
    """Computes the precision@k"""
    batch_size = target.size(0)

    # Positive values are positive labels
    pred = output.ge(0.0).float()

    # To find the correct classifications, we perform an element wise multiplication
    # The result will be 1 only where both inputs are 1
    correct = torch.mul(pred, target.float())

    acc = correct.float().sum() / target.size(1)
    acc *= (1.0 / batch_size)
    return acc


def print_state(idx, epoch, size, batch_time, data_time=None, losses=None, avg_acc=None):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t".format(epoch, idx, size)
    else:
        message = "Test: [{0}/{1}]\t".format(idx, size)

    print(message +
          'Loss {loss.avg:.4f} ({loss.val:.6f})\t'
          'Accuracy {acc.val:.6f} ({acc.avg:.3f})'.format(loss=losses, acc=avg_acc))


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    # switch to train mode
    model = model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.float().cuda()
        x_var = x.cuda()

        # compute output
        output = model(x_var)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.data.item(), x.size(0))
        avg_acc.update(acc, x.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_state(i, epoch, len(train_loader),
                        batch_time, data_time, losses, avg_acc)


def validate(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    # switch to evaluate mode
    model = model.eval()

    end = time.time()
    for i, (x, target) in enumerate(val_loader):
        target = target.float().cuda()
        x = x.cuda()

        # compute output
        output = model(x)
        loss = criterion(output, target)

        # log loss to Pastalog
        # log_val.post('val_loss', value=loss.data[0], step=epoch * len(val_loader) + i)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.data.item(), x.size(0))
        avg_acc.update(acc, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_state(i, -1, len(val_loader),
                        batch_time=batch_time,
                        losses=losses,
                        avg_acc=avg_acc)

    print(' * Prec {acc.avg:.3f}'
          .format(acc=avg_acc))

    return avg_acc.avg
