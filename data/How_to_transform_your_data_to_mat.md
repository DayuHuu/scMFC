# How to transform your data to mat

This guide outlines a pipeline for processing single-cell sequencing data from public databases and converting them into `.mat` format.

**Workflow:**

1.  **Data Collection**
2.  **Normalization** (R / Seurat)
3.  **Feature Selection** (Python / Scanpy)
4.  **Conversion to .mat** (MATLAB)

-----

## 1\. Data Collection

Obtain your data from public repositories (e.g., GEO, Single Cell Portal).

  * **Input Format used here:** Count matrix (`example_counts.csv`) and Metadata (`example_meta.csv`).

-----

## 2\. Data Normalization (R)

Filters cells, log-normalizes data, and separates labels.

**Script:**

```r
library(Seurat)

# 1. Load Data
# Ensure row.names=1 implies gene names are the first column
meta <- read.csv('data/example_meta.csv', header = TRUE)
data <- read.csv('data/example_counts.csv', header = TRUE, row.names = 1)

# 2. Filter (Optional: Select specific cell types or quality control)
# meta <- subset(meta, Type %in% c('TypeA', 'TypeB'))
data <- data[, meta$Cell_ID] # Align data columns with metadata rows

# 3. Normalize
dt <- CreateSeuratObject(counts = data, min.cells = 0, min.features = 0)
dt <- NormalizeData(dt, normalization.method = "LogNormalize", scale.factor = 10000)
dt <- GetAssayData(dt, layer = 'data') 

# 4. Save Outputs
# Save normalized matrix (Genes x Cells -> Transpose to Cells x Genes for Python/MATLAB)
write.csv(as.matrix(t(dt)), 'data/example_norm.csv')

# Save Labels (Adjust column index as needed for your specific label)
write.csv(meta[, c("Cell_ID", "Cell_Type")], 'data/example_labels.csv', row.names = FALSE)
```

-----

## 3\. Highly Variable Genes Selection (Python)

Selects top 2000 highly variable genes (HVGs) using Scanpy.

**Script:**

```python
import scanpy as sc
import os

input_file = 'data/example_norm.csv'
output_file = 'data/example_hvg.csv'

# 1. Read Data
adata = sc.read_csv(input_file)

# 2. Process (HVG Selection)
adata.var_names_make_unique()
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# 3. Save
adata.to_df().to_csv(output_file)
print(f"Saved HVG data to {output_file}")
```

-----

## 4\. Construct .mat File (MATLAB)

Combines the processed data and labels into a `.mat` file.
*Optimization: Used `unique` function to vectorize label mapping.*

**Script:**

```matlab
% Paths
data_path = 'data/example_hvg.csv';
label_path = 'data/example_labels.csv';
save_path = 'data/example.mat';

% 1. Read Data (X) - Skip header row and index column
X = readmatrix(data_path, 'Range', 'B2');

% 2. Read Labels (Y)
raw_labels = readmatrix(label_path, 'Range', 'B2', 'OutputType', 'string');

% 3. Map Labels to Numeric IDs (Vectorized)
[~, ~, Y] = unique(raw_labels); 

% 4. Save
save(save_path, 'X', 'Y');
disp(['Saved: ' save_path]);
```

-----

## Appendix: Handling Multimodal Data

Efficient snippets for converting H5 or MTX files to `.mat`.

### Scenario A: H5 to .mat (Multi-view)

For datasets containing multiple modalities (e.g., RNA + ADT).

```matlab
h5_file = 'data/example_multimodal.h5';

% Read and transpose data to ensure (Cells x Features)
X = cell(2, 1);
X{1} = h5read(h5_file, '/X1')'; 
X{2} = h5read(h5_file, '/X2')'; 

% Read labels
Y = h5read(h5_file, '/Y');

save('data/example_multimodal.mat', 'X', 'Y');
```

### Scenario B: MTX (Sparse) to .mat

*Optimization: Used `sparse` function to avoid slow loops and high memory usage.*

```matlab
% Read MTX format (Row, Col, Value)
fid = fopen('data/example_sparse.mtx', 'r');
data = textscan(fid, '%d %d %f');
fclose(fid);

% Create Sparse Matrix directly (indices, values, dimensions)
rows = data{2}; % Adjust based on file (Gene vs Cell)
cols = data{1};
vals = data{3};

% Create dense matrix only if necessary, otherwise keep as sparse
feature_matrix = full(sparse(rows, cols, vals)); 

save('data/example_from_mtx.mat', 'feature_matrix');
```
