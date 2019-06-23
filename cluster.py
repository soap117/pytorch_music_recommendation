from sklearn.cluster import KMeans
import numpy as np

def eval_threshold(eval_embeddings, threshold, centers_):
    for emb in eval_embeddings:
        min_dis = 1000
        for cen in centers_:
            dis = np.mean(np.square(emb-cen))
            if dis < min_dis:
                min_dis = dis
        if min_dis > threshold:
            return False
    return True
# clustering the embeddings using Kmeans and estimate a threshold by binary search
embeddings = np.load('embeddings_favor.npy')
np.random.shuffle(embeddings)
nums = len(embeddings)
# 85% for clustering 15% for estimating threshold
cluster_nums = int(nums*0.85)
cluster_embeddings = embeddings[:cluster_nums]
eval_embeddings = embeddings[cluster_nums:]
db = KMeans(8).fit(cluster_embeddings)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
centers_ = db.cluster_centers_
inertia_ = db.inertia_
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of inertia: %d' % inertia_)
print(centers_)
np.save('centers.npy', centers_)
left = 0.0
right = 4.0
while right-left>1e-5:
    mid = (left+right)/2
    if eval_threshold(eval_embeddings, mid, centers_):
        right = mid
    else:
        left = mid
print('Estimated number of threshold: %f' % left)
