from sklearn.manifold import TSNE
import numpy as np

import matplotlib.pyplot as plt

def plot_embeddings(word_colour_pairs, corpus_preprocessor, word_embedder, **kwargs):
    words = list(word_colour_pairs.keys())
    embeddings = np.array([word_embedder.embed(corpus_preprocessor.encode_word(w[0]).view(-1,1,26).float().cuda()).cpu().view(-1).numpy() for w in words])
    tsne_embeddings = TSNE(**kwargs).fit_transform(embeddings)

    x_min, x_max = np.min(tsne_embeddings[:, 0]), np.max(tsne_embeddings[:, 0])
    y_min, y_max = np.min(tsne_embeddings[:, 1]), np.max(tsne_embeddings[:, 1])
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    for i, w in enumerate(words):
        x = (tsne_embeddings[i, 0] - x_min) / (x_max - x_min)
        y = (tsne_embeddings[i, 1] - y_min) / (y_max - y_min)
        c = plt.cm.tab20(word_colour_pairs[w])
        plt.text(x, y, w,
                 color=c,
                 fontdict={'weight': 'bold', 'size': 9})