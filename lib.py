
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



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

# Filtering out samples with low intensities of emotions. 
# (If all readers' intensities max )
is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]

df_wrime_target = df_wrime_target.reset_index(drop=True) # これは共通


# Permutation of cluster IDs.
map_of_permutation_for_cluster_IDs = {
    "pretrained": [3, 6, 4, 5, 2, 1], 
    "fine_tuned": [3, 5, 1, 6, 2, 4], 
    "word2vec": [3, 4, 5, 1, 6, 2] 
}



# Make a color map. 
cmap_name = 'gist_rainbow'
cmap = plt.get_cmap(cmap_name)

def appy_dimensionality_reduction(df_wrime_features, clusters, emotion_clusters):
    mappings = []

    from sklearn.decomposition import PCA

    # PCA
    pca = PCA(n_components=2)
    pca.fit(df_wrime_features)
    df_wrime_features_pca = pca.transform(df_wrime_features)
    mappings.append(df_wrime_features_pca) 

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
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        # plt.title(f'UMAP (k={k})')
        plt.colorbar()
        plt.show()

        # plot of embeddings with intensity-based cluster labels.
        plt.figure(figsize=(8, 6))
        plt.scatter(mapping[:, 0], mapping[:, 1], c=emotion_clusters, cmap=cmap_name, alpha=0.7)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        # plt.title(f't-SNE (k={len(set(emotion_clusters))})')
        plt.colorbar()
        plt.show()

        
    return df_wrime_features_pca, df_wrime_features_tsne, df_wrime_features_umap
# end of appy_dimensionality_reduction



def make_embeddings_by_bert(sentences, tokenizer, model, path_to_embeddings):
    import torch
    from transformers import TRANSFORMERS_CACHE
    print(TRANSFORMERS_CACHE)
    from torch.utils.data import DataLoader

    def tokenize(text): # tokenizer function
        return tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    batch_size = 64
    dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=tokenize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Obtain embeddings using batch processing.
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            # move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            hidden_states = outputs.hidden_states
            embeddings.append(hidden_states[-1][:, 0]) # CLS token in the last layer

    embeddings = torch.cat(embeddings)
    # in 30 sec.

    # transform to dataframe object.
    df_embeddings = pd.DataFrame(embeddings.tolist())
    df_embeddings.info()
    # Save.
    df_embeddings.to_csv(path_to_embeddings, sep='\t', index=False, header=False)
    return df_embeddings


def make_embeddings_by_word2vec(sentences, path_to_embeddings):
    import MeCab

    # MeCabのインスタンスを作成
    mecab = MeCab.Tagger()

    # 形態素解析して単語のリストを取得する関数
    def tokenize(text):
        node = mecab.parseToNode(text)
        tokens = []
        while node:
            if node.surface:
                tokens.append(node.surface)
            node = node.next
        return tokens

    # def tokenize2(text):
    #     mecab = MeCab.Tagger("-Owakati")  # 分かち書きのオプションを追加
    #     node = mecab.parseToNode(text)
    #     tokens = []
    #     while node:
    #         if node.surface:
    #             features = node.feature.split(",")  # node.featureから情報を取得
    #             base_form = features[-3] if len(features) > 7 else node.surface  # 基本形を取得
    #             tokens.append(base_form)
    #         node = node.next
    #     return tokens

    tokens1 = tokenize(sentences[1])
    print(tokens1)

    # tokens2 = tokenize2(sentences[1])
    # print(tokens2)

    vector_size = 768

    from gensim.models import word2vec
    # Word2Vecの入力を作成
    data = [tokenize(sentence) for sentence in sentences] # tokenizeの方が良い

    # A bar graph was generated here. 

    # word2vecモデルの訓練
    model = word2vec.Word2Vec(data, vector_size=vector_size, window=5, min_count=1, workers=16, epochs=1000, sample=1e-4, negative=5, sg=1) # sg=0(cbow), sg=1(skip-gram)
    # 3min

    # model.wv.most_similar('すごい', topn=5)
    # number of words. 
    len(model.wv.index_to_key) # tokenize:26633 tokenized2:24361

    import numpy as np
    embeddings = np.array([np.mean([model.wv[token] for token in sentence], axis=0) for sentence in data]) 
    # The embedding of a tweet is formulated as the average of the embeddings of the word 
    # included in the tweet. 

    # To-Do: devise another ways. 

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.info()
    df_embeddings.to_csv(path_to_embeddings, sep='\t', index=False, header=False)
    return df_embeddings



