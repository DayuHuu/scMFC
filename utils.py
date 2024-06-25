import torch
import random
import torch.nn as nn
import faiss
import mkl
from sklearn.cluster import KMeans
from tqdm import tqdm
from network import FCMNet
from torch.utils.data import DataLoader
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import h5py
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mkl.get_max_threads()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def  train_kmeans(x, num_clusters=10, num_gpus=1):
    """
    Runs k-means clustering on one or several GPUs
    """
    d = x.shape[1]
    kmeans = faiss.Clustering(d, num_clusters)
    kmeans.verbose = False
    kmeans.niter = 20

    kmeans.max_points_per_centroid = 100000
    res = [faiss.StandardGpuResources() for i in range(num_gpus)]

    flat_config = []
    for i in range(num_gpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if num_gpus == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(num_gpus)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    kmeans.train(x, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids)

    return centroids.reshape(num_clusters, d)

def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()

def do_clustering(features, num_clusters, num_gpus=None):
    if num_gpus is None:
        num_gpus = faiss.get_num_gpus()
    features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
    centroids = train_kmeans(features, num_clusters, num_gpus)
    labels = compute_cluster_assignment(centroids, features)
    return labels

def extract_features(train_loader, model, device):
    model.eval()
    commonZ = []

    with torch.no_grad():
        for batch_idx, (xs, _) in enumerate(train_loader):
            for v in range(2):
                xs[v] = torch.squeeze(xs[v]).to(device)
            common = model.test_commonZ(xs)
            commonZ.extend(common.detach().cpu().numpy().tolist())
    commonZ = np.array(commonZ)
    return commonZ


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    CUDA_LAUNCH_BLOCKING = 1
    np.random.seed(seed)
    random.seed(seed)

def unsupervised_clustering_step(model, train_loader, labels_holder, n_clusters, device):
    features = extract_features(train_loader, model, device)

    if 'labels' in labels_holder:
        labels_holder['labels_prev_step'] = labels_holder['labels']

    if 'score' not in labels_holder:
        labels_holder['score'] = -1

    labels = do_clustering(features, n_clusters)

    labels_holder['labels'] = labels
    nmi = 0
    nmi_gt = normalized_mutual_info_score(labels_holder['labels_gt'], labels)
    print('NMI t / GT = {:.4f}'.format(nmi_gt))

    if 'labels_prev_step' in labels_holder:
        nmi = normalized_mutual_info_score(labels_holder['labels_prev_step'], labels)
        print('NMI t / t-1 = {:.4f}'.format(nmi))

    train_loader = create_data_loader(train_loader.dataset, train_loader.batch_size, labels=labels)

    return train_loader, nmi_gt, nmi

def create_data_loader(datasets, batch_size, init=False, labels=None):
    if init:
        return DataLoader(datasets, batch_size=batch_size)
    if labels is not None:
        datasets.Y_list = torch.tensor(labels)
        datasets.need_target = True
        return DataLoader(datasets, batch_size=batch_size)
    else:
        datasets.need_target = False
        return DataLoader(datasets, batch_size=batch_size)

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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def T_student( u, u_mean, ncl):
    s = torch.pow(1 + torch.pow(u.unsqueeze(1) - u_mean, 2), -1)
    q = torch.sum(s, dim=2)
    sum1 = torch.sum(q, dim=1)
    re =  torch.div(q, sum1.unsqueeze(1).repeat(1,ncl))
    return  re

def c_means_cost(u, u_mean, p):
    a = torch.mul(u.unsqueeze(1) - u_mean, u.unsqueeze(1) - u_mean)
    b = torch.mul(a, p.unsqueeze(-1))
    return torch.sum(b)

def FCM_Net_train(epoch_f,model,m,u,ncl):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters())
    for TT in range(0, epoch_f):
        optimizer.zero_grad()
        p = model(u)
        p = torch.pow(p, m)
        u_mean = train_kmeans(u.detach().cpu().numpy(), ncl, num_gpus=1)
        u_mean = torch.tensor(u_mean).cuda()
        loss = c_means_cost(u, u_mean, p)
        loss.backward()
        optimizer.step()
    return model

def train_unsupervised(train_loader, model, optimizer, epoch, max_steps, device, tag='unsupervised', verbose=1, num_clusters=16,t1=0.5):
    losses = AverageMeter()

    model.train()
    if verbose == 1:
        pbar = tqdm(total=len(train_loader),
                    ncols=0, desc=f'[{tag.upper()}]', unit=" batch")
    for batch_idx, (xs, _) in enumerate(train_loader):
        for v in range(2):
            xs[v] = torch.squeeze(xs[v]).to(device)
        bs = len(xs[0])
        loss1 = model.get_loss(xs)
        u = model.test_commonZ(xs)
        u_mean = train_kmeans(u.detach().cpu().numpy(), num_clusters, num_gpus=1)
        u_mean = torch.tensor(u_mean).cuda()
        q = T_student(u, u_mean,num_clusters)
        FCM_Net = FCMNet(ncl=num_clusters).cuda()
        FCM_Net = FCM_Net_train(10,FCM_Net,1.5,u.detach(),num_clusters)
        p = FCM_Net(u)
        loss2 = nn.KLDivLoss()(q.log(), p)
        loss = 2*(1-t1)*loss1 + 2*(t1)*loss2
        losses.update(loss.item(), xs[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if verbose == 1:
            pbar.update()
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                epoch=epoch + 1,
                max_steps=max_steps
            )
    if verbose == 1:
        pbar.close()

    return losses.avg

def train(train_loader, model, optimizer, epoch, max_steps, device, tag='train', verbose=1):
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    if verbose == 1:
        pbar = tqdm(total=len(train_loader), ncols=0, desc=f'[{tag.upper()}]', unit=" batch")
    for batch_idx, (Xs, y) in enumerate(train_loader):
        for v in range(2):
            Xs[v] = torch.squeeze(Xs[v]).to(device)

        target = y[0].to(device)
        target = target.to(torch.int64)
        outputs = model(Xs)
        loss = model.get_loss(Xs, target)

        prec1 = accuracy(outputs, target, topk=(1,))[0]  # returns tensors!
        losses.update(loss.item(), Xs[0].size(0))
        acc.update(prec1.item(), Xs[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if verbose == 1:
            pbar.update()
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                Acc=f"{acc.avg:.4f}",
                epoch=epoch + 1,
                max_steps=max_steps
            )

    if verbose == 1:
        pbar.close()

    return acc.avg, losses.avg

def validate(data_loader, model, labels_holder, n_clusters, device):
    commonZ = extract_features(data_loader, model, device)
    acc, nmi, pur, ari = RunKmeans(commonZ, labels_holder['labels_gt'], K=n_clusters, cv=1)
    return acc, nmi, pur, ari
def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
def measure_cluster(y_pred, y_true):
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    cm = confusion_matrix(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    row_max = cm.max(axis=1).sum()
    total = cm.sum()
    pur = row_max / total
    print(f"ARI: {ari * 100:.2f}% NMI: {nmi * 100:.2f}% Acc.: {acc * 100:.2f}% purity: {pur * 100:.2f}%")
    return acc, nmi, pur, ari

def RunKmeans(X, y, K, cv=5):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    results = []
    for _ in range(cv):
        kmeans = KMeans(n_clusters=K)
        kmeans .fit(X)
        y_pred = kmeans .labels_
        results.append(measure_cluster(y_pred, y))
    results = np.array(results).mean(axis=0)
    return results


def load_data(dataset,path):
    data = h5py.File(path + dataset + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)
    np.random.seed(1)
    size = len(Y[0])
    view_num = len(X)

    ##小trick
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    for v in range(view_num):
        X[v] = torch.from_numpy(X[v])

    return X, Y


