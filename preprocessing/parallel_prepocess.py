import cudf
import numba
from numba import cuda

arxiv = cudf.read_csv('./arxiv_normalized_corpus.csv')

arxiv['Feature'] = arxiv['Title'] + ' ' + arxiv['Abstract']
arxiv = arxiv.rename(columns={'Target': 'Section',})
arxiv = arxiv.drop(columns=['Title', 'Abstract'])

arxiv.to_csv('../modeling/arxiv_preprocessed_2.csv', index=False)