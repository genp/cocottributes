import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import logging

logger = logging.getLogger("attributes")
logger.setLevel(logging.INFO)


def get_model(defn):
    model, params = get_network(defn['arch'], defn['n_attrs'],
                                defn['checkpoint'], defn['pretrained'])
    criterion = get_criterion(defn['loss_type'])
    optimizer = get_optimizer(params, defn['fc_lr'], defn['opt_type'])
    return model, criterion, optimizer


def get_network(arch, n_attrs, checkpoint=None, pretrained=True, base_frozen=False):
    """
    Get the network architecture model and the dict of params for passing to the optimizer.
    :param arch:
    :param n_attrs:
    :param checkpoint:
    :param pretrained:
    :param base_frozen:
    :return:
    """
    # We always finetune the model
    logger.info("=> using pre-trained model '{}'".format(arch))
    model = models.__dict__[arch](pretrained=pretrained)

    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        modules = list(model.classifier.children())
        modules.pop()
        modules.append(torch.nn.Linear(4096, n_attrs))
        model.classifier = torch.nn.Sequential(*modules)

    elif arch.startswith("inception") or arch.startswith("google"):
        model.fc = torch.nn.Linear(model.fc.in_features, n_attrs)
        model.AuxLogits.fc = torch.nn.Linear(
            model.AuxLogits.fc.in_features, n_attrs)
        # model = torch.nn.DataParallel(model)

    elif arch.startswith("resnet"):
        model.fc = torch.nn.Linear(model.fc.in_features, n_attrs)

    if checkpoint:
        logger.info("Loading checkpoint: {0}".format(checkpoint))
        state = torch.load(checkpoint)
        model.load_state_dict(state["state_dict"])

    model = model.cuda()

    params = get_params(arch, model, base_frozen=base_frozen)

    return model, params


def get_criterion(loss_type=None, weights=None, args=None):
    logger.info("Using {0} criterion".format(loss_type))
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss().cuda()
    elif loss_type == "MultiLabelSoftMarginLoss":
        return nn.MultiLabelSoftMarginLoss().cuda()


def get_optimizer(params, fc_lr=1e-05, opt_type='SGD', momentum=0.9, weight_decay=1e-4):
    logger.info("Using {0} optimizer".format(opt_type))
    if opt_type == "SGD":
        return optim.SGD(params, lr=fc_lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_type == 'Adam':
        return optim.Adam(params, lr=fc_lr, weight_decay=weight_decay)


def get_params(arch, model, fc_lr=1e-05, base_frozen=False):
    """
    Return the list of params to optimize in the criterion
    :return:
    """
    if arch.startswith("alexnet") or arch.startswith("vgg"):
        base_params_dict = {'params': model.features.parameters()}
        if base_frozen:
            base_params_dict['lr'] = 0.0

        # the classifier needs to learn weights faster
        classifier_params_dict = {
            'params': model.classifier.parameters(), 'lr': fc_lr * 10}

    elif arch.startswith("resnet"):
        ignored_params = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())

        base_params_dict = {'params': base_params}
        if base_frozen:
            base_params_dict['lr'] = 0.0

        classifier_params_dict = {
            'params': model.fc.parameters(), 'lr': fc_lr * 10}

    elif arch.startswith("inception") or arch.startswith("google"):
        fc1_params = list(map(id, model.AuxLogits.fc.parameters()))
        fc2_params = list(map(id, model.fc.parameters()))
        ignored_params = fc1_params + fc2_params
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())

        base_params_dict = {'params': base_params}
        if base_frozen:
            base_params_dict['lr'] = 0.0

        classifier_params_dict = {
            'params': model.fc.parameters(), 'lr': fc_lr * 10}

    else:
        raise RuntimeError('Invalid model architecture')

    params = [base_params_dict, classifier_params_dict]

    return params
