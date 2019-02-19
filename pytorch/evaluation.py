import os
import os.path as osp
import numpy as np
from sklearn.externals import joblib
import logging
from tqdm import tqdm
import torch
from metrics import average_precision

logging.basicConfig(level=logging.INFO)


class Evaluator:
    def __init__(self, model, dataloader,
                 num_labels=204,
                 batch_size=32, name="attributes",
                 train_set=False, verbose=True):
        super().__init__()
        self.logger = logging.getLogger("torch_eval")
        self.logger.setLevel(logging.INFO if verbose else logging.DEBUG)
        self.model = model
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.name = name
        self.train_set = train_set
        logging.info("Batch size: {0}".format(batch_size))

        self.dataloader = dataloader

    def evaluate(self):
        # Array of size the same as the number of labels
        ap = np.zeros((self.num_labels,))
        baseline_ap = np.zeros((self.num_labels,))
        size_counter = 0

        print("# of batches: ", len(self.dataloader))

        ground_truth = np.empty((0, self.num_labels))
        predictions = np.empty((0, self.num_labels))

        # Switch to evaluation mode
        self.model.eval()

        with tqdm(total=len(self.dataloader)) as progress_bar:
            for i, (inp, target) in enumerate(self.dataloader):
                progress_bar.update(1)

                # Load CPU version of target as numpy array
                gts = target.numpy()

                input_var = inp.cuda()
                # compute output
                output = self.model(input_var)
                ests = torch.sigmoid(output).data.cpu().numpy()

                predictions = np.vstack((predictions, ests))
                ground_truth = np.vstack((ground_truth, gts))

                size_counter += ests.shape[0]

        for dim in range(self.num_labels):
            # rescale ground truth to [-1, 1]
            gt = 2*ground_truth[:, dim]-1
            est = predictions[:, dim]
            est_base = np.zeros(est.shape)

            ap_score = average_precision(gt, est)
            base_ap_score = average_precision(gt, est_base)
            ap[dim] = ap_score
            baseline_ap[dim] = base_ap_score

        # for i in range(self.num_labels):
        #     print(ap[i])

        ap_scores = {'ap': ap, 'baseline_ap': baseline_ap,
                     'predictions': predictions, 'ground_truth': ground_truth}
        print('*** mAP and Baseline AP scores ***')
        print(np.mean([a if not np.isnan(a) else 0 for a in ap]))
        print(np.mean([a if not np.isnan(a) else 0 for a in baseline_ap]))

        joblib.dump(ap_scores,
                    osp.join("{0}_ap_scores.jbl".format(self.name)),
                    compress=6)
