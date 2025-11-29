import argparse
import logging
import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from datasets import TrainDataset
from network import Network
from Nmetrics import evaluate
from utils import (
    seed_all,
    load_data,
    create_data_loader,
    train_unsupervised,
    unsupervised_clustering_step,
    train,
    validate,
    extract_features,
    fuse_high_order_embeddings,  
)

import warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train neural network models on a specified dataset."
    )
    parser.add_argument("--dataset", default="BMNC", type=str, required=False)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--epochs", default=1000, type=int)#1000
    parser.add_argument("--K", default=27, type=int, required=False)
    parser.add_argument("--H", default=10, type=int, required=False)
    parser.add_argument(
        "--view_dims", default=[1000, 25], type=int, nargs=2, required=False
    )
    parser.add_argument("--max_unsupervised_steps", default=3, type=int)
    parser.add_argument("--max_supervised_steps", default=2, type=int)
    parser.add_argument("--CVFN", default=1, type=int)
    parser.add_argument("--lam", default=0.5, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data_dir = "../data/"

    try:
        X, Y = load_data(args.dataset, data_dir)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    y_true = Y[0].copy()
    dataset = TrainDataset(X, y_true)
    train_loader = create_data_loader(datasets=dataset, batch_size=args.batch_size)
    labels_holder = {"labels_gt": np.array(y_true)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = Network(
        args.view_dims[0],
        args.view_dims[1],
        args.H,
        K=args.K,
        CVFN=args.CVFN,
    ).to(device)

    cls_optimizer = model.get_cls_optimizer(lr=1e-1)
    recon_optimizer = model.get_recon_optimizer(lr=1e-1)

    best_score = -1.0
    best_epoch = -1
    model_dir = "./checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.dataset}_best_model.pth")

    validate_epoch = 1

    for epoch in tqdm(range(args.epochs), desc="Training"):
        for u_epoch in range(args.max_unsupervised_steps):
            _ = train_unsupervised(
                train_loader,
                model,
                recon_optimizer,
                u_epoch,
                args.max_unsupervised_steps,
                device,
                num_clusters=args.K,
                t1=args.lam,
            )

        if epoch == 0 or epoch % 1 == 0:
            train_loader, nmi_gt, nmi_t_1 = unsupervised_clustering_step(
                model,
                train_loader,
                labels_holder,
                args.K,
                device,
            )

        for u_epoch in range(args.max_supervised_steps):
            _,  _  = train(
                train_loader,
                model,
                cls_optimizer,
                u_epoch,
                args.max_supervised_steps,
                device,
            )

        if (epoch + 1) % validate_epoch == 0:
            acc, nmi, pur, ari = validate(
                train_loader,
                model,
                labels_holder,
                args.K,
                device,
            )

            logging.info(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}, ARI={ari:.4f}"
            )

            if ari > best_score:
                best_score = ari
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                logging.info(
                    f"New best model saved at epoch {epoch + 1} with ARI={ari:.4f}"
                )

    logging.info(
        f"Training finished. Best ARI={best_score:.4f} at epoch {best_epoch + 1}"
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Loaded best model from: {model_path}")
    else:
        logging.warning(
            f"Best model file not found at {model_path}, using last epoch model."
        )

    model.eval()

    common_z = extract_features(train_loader, model, device)

    fused_embeddings = fuse_high_order_embeddings(
        common_z,
        emb1_path="./high/high_v1.emb",
        emb2_path="./high/high_v2.emb",
    )

    kmeans = KMeans(n_clusters=args.K, random_state=1)
    pred = kmeans.fit_predict(fused_embeddings)

    acc, nmi, pur,  _,  _ ,  _ , ari = evaluate(y_true, pred)

    acc = float(np.round(acc, 4))
    nmi = float(np.round(nmi, 4))
    pur = float(np.round(pur, 4))
    ari = float(np.round(ari, 4))

    print(
        "After High-order Fusion",
        "ARI=%.4f, NMI=%.4f, ACC=%.4f, PUR=%.4f" % (ari, nmi, acc, pur),
    )


if __name__ == "__main__":
    seed_all(1)
    main()
