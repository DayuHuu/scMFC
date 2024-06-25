# scMFC
A deep single-cell multi-view fuzzy clustering framework
![Franework](https://github.com/DayuHuu/scMFC/blob/master/scMFC_framework.png)
**Description:**

scEGG is a deep clustering framework designed for single-cell analysis. It integrates cell and exogenous gene features simultaneously, aligning and fusing them during clustering to generate a more discriminative representation.

**Requirements:**

- Python==3.7.0
- Pandas==1.1.5
- Torch==1.13.1
- NumPy==1.21.6
- SciPy==1.7.3
- Scikit-learn==0.22.2

**Datasets:**

- Darmanis: [PubMed](https://pubmed.ncbi.nlm.nih.gov/26060301/)
- Bjorklund: [GSE70580](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580)
- Sun: [GSE128066](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066)
- Marques: [PubMed](https://pubmed.ncbi.nlm.nih.gov/30078729/)
- Zeisel: [PubMed](https://pubmed.ncbi.nlm.nih.gov/25700174/)
- Fink: [PubMed](https://pubmed.ncbi.nlm.nih.gov/35914526/)

**Examples:**

```python
parser.add_argument('--dataset_str', default='Bjorklund', type=str, help='name of dataset')
parser.add_argument('--n_clusters', default=4, type=int, help='expected number of clusters')

# Add other arguments as needed...


```
**Implement:**
```python
python run_scEGG.py
```



