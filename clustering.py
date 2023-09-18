import numpy as np
import collections
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import time


def do_kmeans(clusterNum, vecs, with_center = False):
    result = KMeans(n_clusters=clusterNum, init="k-means++",
                    random_state=20210413).fit_predict(vecs)
    element_num_in_cluster = collections.Counter(list(result))
    if with_center:
        return list(result), element_num_in_cluster, kmeans.cluster_centers_
    else:
        return list(result), element_num_in_cluster

def do_DBSCAN(vec: "ndarray(shape=(N,M))", eps=0.5, minPts=5) -> "cluster":
    """
    クラスタリング手法 DBSCAN
    eps : 半径
    minPts : 近傍点数の閾値
    """
    dbscan = DBSCAN(eps=eps, min_samples=minPts).fit(vec)
    labels = dbscan.labels_
    which_method = labels + 1  # "+1"は"-1"というクラスタをなくすため
    cluster_num = np.max(which_method)  # outlierもクラスタの1つと考える

    return which_method, cluster_num


def tuning_DBSCAN(vec: "ndarray(shape=(N,M))", seek_clusterNum_min, seek_clusterNum_max):
    """
    DBSCANのハイパーパラメータ(eps, minPts)を決めるための参考に
    """
    for eps in range(1, 50, 1):
        eps = eps / 10

        for minPts in range(1, 20):
            dbscan, clusterNum = do_DBSCAN(vec, eps, minPts)

            if seek_clusterNum_min <= clusterNum and clusterNum <= seek_clusterNum_max:
                print("<", "eps:", eps, ",minPts:", minPts, ">")
                print("\tcluster num : \t", clusterNum)  # クラスタ数
                print("\toutlier num : \t", len(
                    np.where(dbscan == 0)[0]))  # outlier数
                print("\tdata point num in cluster1 : \t", len(
                    np.where(dbscan == 1)[0]))  # クラスタ1に含まれる点の数
                print("\tdata point num in cluster2 : \t", len(
                    np.where(dbscan == 2)[0]))  # クラスタ2に含まれる点の数
                print("\tdata point num in cluster3 : \t", len(
                    np.where(dbscan == 3)[0]))  # クラスタ3に含まれる点の数
                print("\tdata point num in cluster4 : \t", len(
                    np.where(dbscan == 4)[0]))  # クラスタ4に含まれる点の数


def do_MeanShift(vec, bandwidth):
    meanshift = MeanShift(bandwidth=bandwidth).fit(vec)
    labels = meanshift.labels_
    cluster_num = np.max(labels) + 1

    return labels, cluster_num


def do_Hierarchical(vec, n_clusters, affinity, linkage):
    agglomerative = AgglomerativeClustering(
        n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = agglomerative.fit_predict(vec)
    cluster_num = np.max(labels) + 1

    return labels, cluster_num

def do_GMM(num_clusters, uas_vectors):
    # Initalize a GMM object and use it for clustering.
    clf =  GaussianMixture(n_components=num_clusters,
                    covariance_type="tied", init_params='kmeans', max_iter=50)
    # Get cluster assignments.
    clf.fit(uas_vectors)
    idx = clf.predict(uas_vectors)
    start = time.time()
    print ("Clustering Done...", time.time()-start, "seconds")
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(uas_vectors)
    return (idx, idx_proba)