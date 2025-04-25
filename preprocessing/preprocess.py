import pandas as pd

arxiv = pd.read_csv('./arxiv_normalized_corpus.csv')

arxiv['Feature'] = arxiv['Title'] + ' ' + arxiv['Abstract']

# rename a column
arxiv = arxiv.rename(columns={'Target': 'Section',})

arxiv = arxiv.drop(columns=['Title', 'Abstract'])

print(arxiv.head())

# save the dataframe to a csv file
arxiv.to_csv('../modeling/arxiv_preprocessed.csv', index=False)
print("Preprocessing complete. Data saved to /modeling/arxiv_preprocessed.csv")