import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from datasets import TrainDataset
from network import Network
from utils import seed_all, load_data, create_data_loader, train_unsupervised, unsupervised_clustering_step, train, validate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train neural network models on specified dataset.')
    parser.add_argument('--dataset', default='BMNC',type=str, required=False, help='Dataset to use, default is BMNC.')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size for training.')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs to train.')#1000
    parser.add_argument('--K', default=27,type=int, required=False, help='Number of clusters for KMeans.')
    parser.add_argument('--H', default=10,type=int, required=False, help='Hidden layer size.')
    parser.add_argument('--view_dims', default=[1000,25], type=int, nargs=2, required=False, help='Dimensions of each view for the network.')
    parser.add_argument('--max_unsupervised_steps', default=3, type=int, help='Maximum unsupervised learning steps.')
    parser.add_argument('--max_supervised_steps', default=2, type=int, help='Maximum supervised learning steps.')
    parser.add_argument('--CVFN', default=1, type=int, help='Flag for Cross-Validation Fold Number.')
    parser.add_argument('--lam', default=0.5, type=float, help='Lambda for regularization.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_dir = './data/'
    try:
        X, Y = load_data(args.dataset, data_dir)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    y_true = Y[0].copy()
    dataset = TrainDataset(X, y_true)
    train_loader = create_data_loader(datasets=dataset, batch_size=args.batch_size)
    labels_holder = {}
    labels_holder['labels_gt'] = np.array(y_true)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network(args.view_dims[0], args.view_dims[1], args.H, K=args.K, CVFN=args.CVFN).to(device)
    cls_optimizer = model.get_cls_optimizer(lr=1e-1)
    recon_optimizer = model.get_recon_optimizer(lr=1e-1)

    validate_epoch = 1
    for epoch in tqdm(range(args.epochs), desc='Training'):
        nmi_gt = None
        for u_epoch in range(args.max_unsupervised_steps):
            loss_avg = train_unsupervised(train_loader, model, recon_optimizer, u_epoch, args.max_unsupervised_steps,
                                          device, num_clusters=args.K,t1=args.lam)
        if epoch == 0 or epoch % 1 == 0:
            train_loader, nmi_gt, nmi_t_1 = \
                unsupervised_clustering_step(model, train_loader, labels_holder, args.K, device)
        for u_epoch in range(args.max_supervised_steps):
            acc_avg, loss_avg = train(train_loader, model, cls_optimizer, u_epoch, args.max_supervised_steps, device,
                                      )
        if (epoch + 1) % validate_epoch == 0:
            acc, nmi, pur, ari = validate(train_loader, model, labels_holder, args.K, device)

if __name__ == '__main__':
    seed_all(1)
    main()
