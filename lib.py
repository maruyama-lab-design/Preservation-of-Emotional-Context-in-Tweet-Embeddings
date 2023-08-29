import pandas as pd

# loading data
df_wrime = pd.read_table('wrime-ver1.tsv')
# df_wrime.info()

print(df_wrime['Avg. Readers_Joy'])

# the eight basic emotions by Plutchik. 
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
# Representations of them in Japanese. 
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼'] 

# The eight "Avg. Readers_*" columns are integrated into a list like [0, 2, 0, 0, 0, 0, 0, 0]. 
df_wrime['readers_emotion_intensities'] = df_wrime.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)
print(df_wrime['readers_emotion_intensities'])

# (If all readers' intensities max )
is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]



import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# Make a color map. 
cmap_name = 'gist_rainbow'
cmap = plt.get_cmap(cmap_name)


import numpy as np


def appy_dimensionality_reduction(df_wrime_features, clusters):
    mappings = []

    from sklearn.decomposition import PCA

    # PCA
    pca = PCA(n_components=2)
    pca.fit(df_wrime_features)
    mappings.append(pca.transform(df_wrime_features)) 

    from sklearn.manifold import TSNE

    # t-SNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=50)
    df_wrime_features_tsne = tsne.fit_transform(df_wrime_features)
    mappings.append(df_wrime_features_tsne)

    import umap

    # UMAP
    umap_obj = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='yule')
    # umap_obj = umap.UMAP(n_components=2, random_state=0) # in pretrained model. 
    df_wrime_features_umap = umap_obj.fit_transform(df_wrime_features)
    mappings.append(df_wrime_features_umap)
    

    for mapping in mappings:
        plt.figure(figsize=(8, 6))
        plt.scatter(mapping[:, 0], mapping[:, 1], c=clusters, cmap=cmap_name, alpha=0.7)
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        # plt.title(f'UMAP (k={k})')
        plt.colorbar()
        plt.show()
        
    return df_wrime_features_tsne, df_wrime_features_umap
# end of appy_dimensionality_reduction








