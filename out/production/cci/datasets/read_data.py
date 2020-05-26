import pandas as pd
from time import time
from scipy.sparse import csr_matrix

root = '../../data/mammary_gland'
train = [3510, 1311, 6633, 6905, 4909, 2081]
test = [1059, 648, 1592]
all_data = train + test


# prepare unified genes
id2gene = []
with open(f'{root}/mammary_gland_gene_label.txt', 'r', encoding='utf-8') as f:
    for line in f:
        id2gene.append(line.strip())
gene2id = {gene: idx for idx, gene in enumerate(id2gene)}

for num in all_data[1:]:
    data_path = f'{root}/mouse_Mammary_gland{num}_data.csv'
    type_path = f'{root}/mouse_Mammary_gland{num}_celltype.csv'
    start = time()
    df = pd.read_csv(data_path, index_col=0)
    df = df.transpose(copy=True)
    df = df.rename(columns=gene2id)
    print(time() - start, 's')
    print(df.shape)
    print(df.head())
    arr = df.to_numpy()
    row_idx, col_idx = arr.nonzero()
    print(row_idx)
    print(col_idx)
    
    src_idx = row_idx + len(df)
    tgt_idx = df.columns[col_idx].to_list()
    info = arr[(row_idx, col_idx)]
    values = csr_matrix((info, (row_idx, col_idx)), shape=df.shape)
    break
    """
    columns = df.columns
    missing_genes = [col for col in id2gene if col not in columns]
    for col in missing_genes:
        df[col] = 0.0
    print(time() - start, 's')

    print(df.shape)
    print(df[id2gene].shape)

    """
