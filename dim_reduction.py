import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pandas import plotting


def do_tSNE(vec: "ndarray(shape=(N,M))", com: int, p = 40, lr = 100, random_state = 20200520) -> "ndarray(shape=(N,2))":
    """
    次元削減手法t-SNE
    comp : 何次元まで削減するか
    """
    tsne = TSNE(n_components=com, perplexity = p,
                learning_rate = 100, random_state=20200520)
    result = tsne.fit_transform(vec)
    return result


def do_PCA(vec: "ndarray(shape=(N,M))", specific=None, do_stadardization=False):
    """
    次元削減手法PCA
    specific : 色を変化させるユーザ
    do_standardization : 次元削減前にデータの標準化を行うかどうか
    """
    if do_stadardization:
        vec = standardization(vec, axis=0)

    pca = PCA()
    feature = pca.fit(vec)
    feature = pca.transform(vec)

    # projection(主成分ベクトルが新しい軸になるように)
    # https://qiita.com/supersaiakujin/items/138c0d8e6511735f1f45

    # 寄与率
    plt.figure(figsize=(100, 50))
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()

    # 特定ユーザのみ色を変えることで，どの主成分を見るのが有効かを見極める
    color = [0 for _ in range(vec.shape[0])]
    if specific != None:
        color = []

        for i in range(vec.shape[0]):
            if i in specific:
                color.append(2)
            else:
                color.append(0)

    # 主成分の全組み合わせのplot
    df_for_pca_feature = pd.DataFrame(feature[:, :30])
    plotting.scatter_matrix(
        df_for_pca_feature, figsize=(20, 20), alpha=0.5, c=color)
    plt.show()

    return feature, pca
