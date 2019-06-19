# Pytorch Music Recommendation System
personal experiment staff
## Motivation
Yet, there exsits many deep learning based music recommendation system, but they usually require supervised signal(favor list and unfavor list) or collaborative filtering, which is impractic for a normal user to apply. 
For a ordinary user, it's most likely that you only have a favor list of songs and a lot of songs you don't know you like or not. Based on above situation, this approach is designed.
## Method
### Music2Vector
The first step is to use unsupervised approaches to vectornize each music into a embedding space. The major idea is similar to Doc2Vec approach. For details please see the below image.
### Clustering
You now have a embedding space for music! Still it's not enough to build a recommendation system. In this step, clustering algorithm is adopted to find the cluster centers of your favor songs. Then use those centers and set a threshold, you can now guess if a new song satisfies you according to the minimum distance between one of those centers.
## Does it work?
I do not perform a quantitative research on it. But usually for a 30s music segment, its' cloest neighbors are other segments from the same song or sgements from a different version of that song. The below images shows the projector results of my embedding space.
