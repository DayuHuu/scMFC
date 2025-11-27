import argparse
import warnings
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Local imports
import load_data as loader
from datasets import TrainDataset
from network import Network
from Nmetrics import evaluate
from utils import (
    seed_all,
    create_data_loader,
    train_unsupervised,
    unsupervised_clustering_step,
    train,
    validate,
    extract_features
)

# Configuration
warnings.filterwarnings("ignore")

# Define paths using pathlib for better cross-platform compatibility
RESULTS_DIR = Path("./results")
SAVE_DIR = Path("./testsave")
DATA_DIR = Path("./data")
EMB_DIR = Path("./emb")


def setup_directories():
    """Ensure necessary directories exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_args(data_para):
    """Configure and parse arguments."""
    parser = argparse.ArgumentParser(description='scEGG Main Pipeline')

    # Dataset specific defaults
    parser.add_argument('--dataset', default=data_para, help='Dataset configuration dictionary')
    parser.add_argument("--K", default=data_para['K'], type=int, help='Number of clusters')
    parser.add_argument("--H", default=data_para['n_hid'], help='Hidden layer dimensions')
    parser.add_argument("--view_dims", default=data_para['n_input'], help='Input view dimensions')

    # Training hyperparameters
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--epochs', default=16, type=int)
    parser.add_argument("--feature_dim", default=10, type=int)
    parser.add_argument("--max_unsupervised_steps", default=3, type=int)
    parser.add_argument("--max_supervised_steps", default=2, type=int)
    parser.add_argument("--verbose", default=0, type=int)
    parser.add_argument("--l", default=0.5, type=float, help='Balance parameter lambda')

    # Note: is_high, is_alternate, CVFN are removed and hardcoded in logic as per requirements

    return parser.parse_args()


def train_pipeline(args, train_loader, labels_holder, device, model_path):
    """Executes the alternating training process."""

    # Initialize Model with fixed CVFN=1
    model = Network(
        args.view_dims[0],
        args.view_dims[1],
        args.H[0],
        K=args.K,
        CVFN=1
    ).to(device)

    cls_optimizer = model.get_cls_optimizer(lr=1e-1)
    recon_optimizer = model.get_recon_optimizer(lr=1e-1)

    best_score = 0
    validate_epoch = 1
    epoch_save = 0

    # Training Loop
    t_progress = tqdm(range(args.epochs), desc='Training', unit='epoch')
    for epoch in t_progress:
        # 1. Unsupervised Phase
        for u_epoch in range(args.max_unsupervised_steps):
            train_unsupervised(
                train_loader, model, recon_optimizer, u_epoch,
                args.max_unsupervised_steps, device,
                verbose=args.verbose, num_clusters=args.K, t1=args.l
            )

        # 2. Clustering Update / Sync Phase
        if epoch == 0 or epoch % 1 == 0:
            train_loader, _, _ = unsupervised_clustering_step(
                model, train_loader, labels_holder, args.K, device
            )

        # 3. Supervised Phase
        for u_epoch in range(args.max_supervised_steps):
            train(
                train_loader, model, cls_optimizer, u_epoch,
                args.max_supervised_steps, device, verbose=args.verbose
            )

        # 4. Validation Phase
        if (epoch + 1) % validate_epoch == 0:
            _, _, _, ari = validate(train_loader, model, labels_holder, args.K, device)
            if ari > best_score:
                best_score = ari
                epoch_save = epoch
                torch.save(model.state_dict(), model_path)
                t_progress.set_postfix({'Best ARI': f"{best_score:.4f}"})

    return epoch_save


def evaluate_pipeline(args, train_loader, y_true, device, model_path, dataset_name, epoch_save, log_file):
    """Performs final evaluation using best model and gene embeddings."""

    # Reload best model
    model = Network(
        args.view_dims[0],
        args.view_dims[1],
        args.H[0],
        K=args.K,
        CVFN=1
    ).to(device)

    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Helper to calculate metrics
    def calculate_metrics(features, label_prefix):
        kmeans = KMeans(n_clusters=args.K, random_state=1)
        kmeans.fit(features)
        pred = kmeans.labels_

        acc, nmi, pur, _, _, _, ari = evaluate(y_true, pred)
        return {
            'Phase': label_prefix,
            'ARI': float(np.round(ari, 4)),
            'NMI': float(np.round(nmi, 4)),
            'ACC': float(np.round(acc, 4)),
            'PUR': float(np.round(pur, 4))
        }

    # 1. Evaluate "Before" (Model features only)
    commonZ = extract_features(train_loader, model, device)
    commonZ_tensor = torch.tensor(commonZ)

    # Note: Logic follows original code where features are appended to a list then concatenated
    fea_allemb = [commonZ_tensor]
    fea_allemb_before = torch.cat(fea_allemb, dim=-1)

    stats_before = calculate_metrics(fea_allemb_before, 'H')

    # 2. Evaluate "After" (With Embeddings)
    # The original logic keeps `fea_allemb` list populated with commonZ and appends embeddings
    emb_file_1 = EMB_DIR / f"high_{dataset_name}_v1.emb"
    emb_file_2 = EMB_DIR / f"high_{dataset_name}_v2.emb"

    stats_after = None
    if emb_file_1.exists() and emb_file_2.exists():
        try:
            emb1 = torch.tensor(np.loadtxt(emb_file_1, dtype=float))
            emb2 = torch.tensor(np.loadtxt(emb_file_2, dtype=float))

            fea_allemb.append(emb1)
            fea_allemb.append(emb2)

            fea_combined = torch.cat(fea_allemb, dim=-1)
            stats_after = calculate_metrics(fea_combined, 'H')

            print(f"Final Result [After H]: ARI={stats_after['ARI']:.4f}, "
                  f"NMI={stats_after['NMI']:.4f}, ACC={stats_after['ACC']:.4f}")

        except Exception as e:
            print(f"Error loading embeddings: {e}")
    else:
        print(f"Warning: Embedding files not found in {EMB_DIR}. Skipping 'After' evaluation.")

    # Logging to file
    log_file.write(str(args) + '\n')
    log_file.write(f"The best epoch is: {epoch_save}\n")

    # The original code formats the dictionary specifically like this:
    # {'Before': 'H', 'ARI': ..., ...}
    log_dict_before = stats_before.copy()
    log_dict_before['Before'] = log_dict_before.pop('Phase')
    log_file.write(str(log_dict_before) + '\n')

    if stats_after:
        log_dict_after = stats_after.copy()
        log_dict_after['After'] = log_dict_after.pop('Phase')
        log_file.write(str(log_dict_after) + '\n')


def run_experiment(dataset_key, data_para):
    """Runs the full experiment for a single dataset."""
    print(f"{'=' * 20} Running Dataset: {data_para[1]} {'=' * 20}")

    # Setup
    seed_all(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare arguments
    # Note: Argument parsing inside the loop mimics original behavior
    # to allow data_para to set defaults dynamically.
    args = get_args(data_para)

    # Data Loading
    X, Y = loader.load_data(args.dataset, str(DATA_DIR) + '/')
    y_true = Y[0].copy()
    dataset = TrainDataset(X, y_true)
    train_loader = create_data_loader(datasets=dataset, batch_size=args.batch_size, init=True)

    labels_holder = {'labels_gt': np.array(y_true)}

    # Define File Paths
    model_save_path = SAVE_DIR / f"final{data_para[1]}_model.pth"
    result_file_path = RESULTS_DIR / f"{data_para[1]}.txt"

    # Open log file
    with open(result_file_path, 'a+') as f:
        # Train
        best_epoch = train_pipeline(args, train_loader, labels_holder, device, model_save_path)

        # Evaluate
        evaluate_pipeline(
            args, train_loader, y_true, device,
            model_save_path, data_para[2], best_epoch, f
        )


def main():
    setup_directories()

    # Iterate through all datasets defined in load_data.py
    for key, data_para in loader.ALL_data.items():
        try:
            run_experiment(key, data_para)
        except Exception as e:
            print(f"An error occurred while processing {key}: {e}")
            # Optional: raise e # Uncomment to stop on first error


if __name__ == "__main__":
    main()
