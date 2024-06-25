# scMFC
A deep single-cell multi-view fuzzy clustering framework
![Framework](https://github.com/DayuHuu/scMFC/blob/master/scMFC.png)



**Description:**

This is some of source code for **scMFC: High-order Topology for Deep Single-cell Multi-view Fuzzy Clustering.** The first module involves high-order neighborhood enhancement. Initially, we construct the 1-order neighborhood relationship among cells and then conduct random walks on it to generate a high-order enhancement for each cell view, effectively addressing the issue of underutilization of existing neighborhood information. The second module focuses on cross-view information aggregation. We employ a global structure relationship aggregator to dynamically allocate embedding weights across different views, thus effectively addressing the information differences between them. The third module, the Deep Fuzzy Clustering module, employs a deep fully connected network to estimate the actual cluster assignments, which is trained through minimizing a combination of reconstruction loss and clustering loss. Experiments on three real-world single-cell multi-view datasets have demonstrated the stability and superiority of our method.

**Requirements:**
- Python==3.7.0
- Pandas==1.1.5
- Torch==1.13.1
- NumPy==1.21.6
- SciPy==1.7.3
- Scikit-learn==0.22.2

**Datasets:**

- Refer to the data file

**Examples:**

```python
    parser = argparse.ArgumentParser(description='Train neural network models on specified dataset.')
    parser.add_argument('--dataset', default='BMNC',type=str, required=False, help='Dataset to use, default is BMNC.')
    parser.add_argument('--batch_size', default=2048, type=int, help='Batch size for training.')

# Add other arguments as needed...

```
**Implement:**
```python
python run_scEGG.py
```



