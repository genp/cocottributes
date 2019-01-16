import arguments
import os
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sklearn.externals import joblib

from dataset import COCOAttributes
from svm_model import FeatureExtractor, SVM
from tqdm import tqdm
from metrics import average_precision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attributes")


def get_features(feat_extractor, data_loader, n_attrs=204, split="train"):
    feat_extractor = feat_extractor.eval()

    svm_feats_file = os.path.join("svm", "svm_features_{0}.jbl".format(split))

    if os.path.exists(svm_feats_file):
        [feats, targets] = joblib.load(svm_feats_file)
    else:
        with torch.no_grad():
            feats = np.empty((0, feat_extractor.output_dim))
            targets = np.empty((0, n_attrs))

            for i, (x, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
                feat = feat_extractor(x)
                feats = np.vstack((feats, feat))
                targets = np.vstack((targets, target.numpy()))

            joblib.dump([feats, targets], svm_feats_file)

    return feats, targets


def main():
    args = arguments.parse()

    feat_extractor = FeatureExtractor()
    svms = [SVM() for _ in range(args.n_attrs)]

    # Dataloader code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
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

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info("Beginning training...")

    feats, targets = get_features(
        feat_extractor, train_loader, n_attrs=args.n_attrs, split='train')

    if args.checkpoint:
        [svms, _] = joblib.load(args.checkpoint)

    else:
        for i in range(args.n_attrs):
            print("Training for attribute", i)
            # using [0, 1] or [-1, 1] doesn't really make a difference
            svms[i].train(feats, targets[:, i])
            print()

    logger.info('Finished Training')

    logger.info("Running evaluation")

    feats, targets = get_features(
        feat_extractor, val_loader, n_attrs=args.n_attrs, split='val')

    ap_scores = []

    for i in range(args.n_attrs):
        est = svms[i].test(feats)
        ap_score = average_precision(2*targets[:, i]-1, est)
        print("AP score for {0}".format(i), ap_score)
        ap_scores.append(ap_score)

    print("mean AP", sum(ap_scores)/args.n_attrs)

    if not args.checkpoint:
        logger.info("Saving models and AP scores")
        joblib.dump([svms, ap_scores], "svm_baseline.jbl")


if __name__ == "__main__":
    main()
