---
tags:
- mteb
- sentence_embedding
- feature_extraction
- sentence-transformers
- transformers
- transformers.js
model-index:
- name: UAE-Large-V1
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 75.55223880597015
    - type: ap
      value: 38.264070815317794
    - type: f1
      value: 69.40977934769845
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 92.84267499999999
    - type: ap
      value: 89.57568507997713
    - type: f1
      value: 92.82590734337774
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 48.292
    - type: f1
      value: 47.90257816032778
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 42.105
    - type: map_at_10
      value: 58.181000000000004
    - type: map_at_100
      value: 58.653999999999996
    - type: map_at_1000
      value: 58.657000000000004
    - type: map_at_3
      value: 54.386
    - type: map_at_5
      value: 56.757999999999996
    - type: mrr_at_1
      value: 42.745
    - type: mrr_at_10
      value: 58.437
    - type: mrr_at_100
      value: 58.894999999999996
    - type: mrr_at_1000
      value: 58.897999999999996
    - type: mrr_at_3
      value: 54.635
    - type: mrr_at_5
      value: 56.99999999999999
    - type: ndcg_at_1
      value: 42.105
    - type: ndcg_at_10
      value: 66.14999999999999
    - type: ndcg_at_100
      value: 68.048
    - type: ndcg_at_1000
      value: 68.11399999999999
    - type: ndcg_at_3
      value: 58.477000000000004
    - type: ndcg_at_5
      value: 62.768
    - type: precision_at_1
      value: 42.105
    - type: precision_at_10
      value: 9.110999999999999
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 23.447000000000003
    - type: precision_at_5
      value: 16.159000000000002
    - type: recall_at_1
      value: 42.105
    - type: recall_at_10
      value: 91.11
    - type: recall_at_100
      value: 99.14699999999999
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 70.341
    - type: recall_at_5
      value: 80.797
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 49.02580759154173
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 43.093601280163554
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 64.19590406875427
    - type: mrr
      value: 77.09547992788991
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 87.86678362843676
    - type: cos_sim_spearman
      value: 86.1423242570783
    - type: euclidean_pearson
      value: 85.98994198511751
    - type: euclidean_spearman
      value: 86.48209103503942
    - type: manhattan_pearson
      value: 85.6446436316182
    - type: manhattan_spearman
      value: 86.21039809734357
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 87.69155844155844
    - type: f1
      value: 87.68109381943547
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 39.37501687500394
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 37.23401405155885
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 30.232
    - type: map_at_10
      value: 41.404999999999994
    - type: map_at_100
      value: 42.896
    - type: map_at_1000
      value: 43.028
    - type: map_at_3
      value: 37.925
    - type: map_at_5
      value: 39.865
    - type: mrr_at_1
      value: 36.338
    - type: mrr_at_10
      value: 46.969
    - type: mrr_at_100
      value: 47.684
    - type: mrr_at_1000
      value: 47.731
    - type: mrr_at_3
      value: 44.063
    - type: mrr_at_5
      value: 45.908
    - type: ndcg_at_1
      value: 36.338
    - type: ndcg_at_10
      value: 47.887
    - type: ndcg_at_100
      value: 53.357
    - type: ndcg_at_1000
      value: 55.376999999999995
    - type: ndcg_at_3
      value: 42.588
    - type: ndcg_at_5
      value: 45.132
    - type: precision_at_1
      value: 36.338
    - type: precision_at_10
      value: 9.17
    - type: precision_at_100
      value: 1.4909999999999999
    - type: precision_at_1000
      value: 0.196
    - type: precision_at_3
      value: 20.315
    - type: precision_at_5
      value: 14.793000000000001
    - type: recall_at_1
      value: 30.232
    - type: recall_at_10
      value: 60.67399999999999
    - type: recall_at_100
      value: 83.628
    - type: recall_at_1000
      value: 96.209
    - type: recall_at_3
      value: 45.48
    - type: recall_at_5
      value: 52.354
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 32.237
    - type: map_at_10
      value: 42.829
    - type: map_at_100
      value: 44.065
    - type: map_at_1000
      value: 44.199
    - type: map_at_3
      value: 39.885999999999996
    - type: map_at_5
      value: 41.55
    - type: mrr_at_1
      value: 40.064
    - type: mrr_at_10
      value: 48.611
    - type: mrr_at_100
      value: 49.245
    - type: mrr_at_1000
      value: 49.29
    - type: mrr_at_3
      value: 46.561
    - type: mrr_at_5
      value: 47.771
    - type: ndcg_at_1
      value: 40.064
    - type: ndcg_at_10
      value: 48.388
    - type: ndcg_at_100
      value: 52.666999999999994
    - type: ndcg_at_1000
      value: 54.67100000000001
    - type: ndcg_at_3
      value: 44.504
    - type: ndcg_at_5
      value: 46.303
    - type: precision_at_1
      value: 40.064
    - type: precision_at_10
      value: 9.051
    - type: precision_at_100
      value: 1.4500000000000002
    - type: precision_at_1000
      value: 0.193
    - type: precision_at_3
      value: 21.444
    - type: precision_at_5
      value: 15.045
    - type: recall_at_1
      value: 32.237
    - type: recall_at_10
      value: 57.943999999999996
    - type: recall_at_100
      value: 75.98700000000001
    - type: recall_at_1000
      value: 88.453
    - type: recall_at_3
      value: 46.268
    - type: recall_at_5
      value: 51.459999999999994
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 38.797
    - type: map_at_10
      value: 51.263000000000005
    - type: map_at_100
      value: 52.333
    - type: map_at_1000
      value: 52.393
    - type: map_at_3
      value: 47.936
    - type: map_at_5
      value: 49.844
    - type: mrr_at_1
      value: 44.389
    - type: mrr_at_10
      value: 54.601
    - type: mrr_at_100
      value: 55.300000000000004
    - type: mrr_at_1000
      value: 55.333
    - type: mrr_at_3
      value: 52.068999999999996
    - type: mrr_at_5
      value: 53.627
    - type: ndcg_at_1
      value: 44.389
    - type: ndcg_at_10
      value: 57.193000000000005
    - type: ndcg_at_100
      value: 61.307
    - type: ndcg_at_1000
      value: 62.529
    - type: ndcg_at_3
      value: 51.607
    - type: ndcg_at_5
      value: 54.409
    - type: precision_at_1
      value: 44.389
    - type: precision_at_10
      value: 9.26
    - type: precision_at_100
      value: 1.222
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 23.03
    - type: precision_at_5
      value: 15.887
    - type: recall_at_1
      value: 38.797
    - type: recall_at_10
      value: 71.449
    - type: recall_at_100
      value: 88.881
    - type: recall_at_1000
      value: 97.52
    - type: recall_at_3
      value: 56.503
    - type: recall_at_5
      value: 63.392
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.291999999999998
    - type: map_at_10
      value: 35.65
    - type: map_at_100
      value: 36.689
    - type: map_at_1000
      value: 36.753
    - type: map_at_3
      value: 32.995000000000005
    - type: map_at_5
      value: 34.409
    - type: mrr_at_1
      value: 29.04
    - type: mrr_at_10
      value: 37.486000000000004
    - type: mrr_at_100
      value: 38.394
    - type: mrr_at_1000
      value: 38.445
    - type: mrr_at_3
      value: 35.028
    - type: mrr_at_5
      value: 36.305
    - type: ndcg_at_1
      value: 29.04
    - type: ndcg_at_10
      value: 40.613
    - type: ndcg_at_100
      value: 45.733000000000004
    - type: ndcg_at_1000
      value: 47.447
    - type: ndcg_at_3
      value: 35.339999999999996
    - type: ndcg_at_5
      value: 37.706
    - type: precision_at_1
      value: 29.04
    - type: precision_at_10
      value: 6.192
    - type: precision_at_100
      value: 0.9249999999999999
    - type: precision_at_1000
      value: 0.11
    - type: precision_at_3
      value: 14.802000000000001
    - type: precision_at_5
      value: 10.305
    - type: recall_at_1
      value: 27.291999999999998
    - type: recall_at_10
      value: 54.25299999999999
    - type: recall_at_100
      value: 77.773
    - type: recall_at_1000
      value: 90.795
    - type: recall_at_3
      value: 39.731
    - type: recall_at_5
      value: 45.403999999999996
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 18.326
    - type: map_at_10
      value: 26.290999999999997
    - type: map_at_100
      value: 27.456999999999997
    - type: map_at_1000
      value: 27.583000000000002
    - type: map_at_3
      value: 23.578
    - type: map_at_5
      value: 25.113000000000003
    - type: mrr_at_1
      value: 22.637
    - type: mrr_at_10
      value: 31.139
    - type: mrr_at_100
      value: 32.074999999999996
    - type: mrr_at_1000
      value: 32.147
    - type: mrr_at_3
      value: 28.483000000000004
    - type: mrr_at_5
      value: 29.963
    - type: ndcg_at_1
      value: 22.637
    - type: ndcg_at_10
      value: 31.717000000000002
    - type: ndcg_at_100
      value: 37.201
    - type: ndcg_at_1000
      value: 40.088
    - type: ndcg_at_3
      value: 26.686
    - type: ndcg_at_5
      value: 29.076999999999998
    - type: precision_at_1
      value: 22.637
    - type: precision_at_10
      value: 5.7090000000000005
    - type: precision_at_100
      value: 0.979
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 12.894
    - type: precision_at_5
      value: 9.328
    - type: recall_at_1
      value: 18.326
    - type: recall_at_10
      value: 43.824999999999996
    - type: recall_at_100
      value: 67.316
    - type: recall_at_1000
      value: 87.481
    - type: recall_at_3
      value: 29.866999999999997
    - type: recall_at_5
      value: 35.961999999999996
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.875
    - type: map_at_10
      value: 40.458
    - type: map_at_100
      value: 41.772
    - type: map_at_1000
      value: 41.882999999999996
    - type: map_at_3
      value: 37.086999999999996
    - type: map_at_5
      value: 39.153
    - type: mrr_at_1
      value: 36.381
    - type: mrr_at_10
      value: 46.190999999999995
    - type: mrr_at_100
      value: 46.983999999999995
    - type: mrr_at_1000
      value: 47.032000000000004
    - type: mrr_at_3
      value: 43.486999999999995
    - type: mrr_at_5
      value: 45.249
    - type: ndcg_at_1
      value: 36.381
    - type: ndcg_at_10
      value: 46.602
    - type: ndcg_at_100
      value: 51.885999999999996
    - type: ndcg_at_1000
      value: 53.895
    - type: ndcg_at_3
      value: 41.155
    - type: ndcg_at_5
      value: 44.182
    - type: precision_at_1
      value: 36.381
    - type: precision_at_10
      value: 8.402
    - type: precision_at_100
      value: 1.278
    - type: precision_at_1000
      value: 0.16199999999999998
    - type: precision_at_3
      value: 19.346
    - type: precision_at_5
      value: 14.09
    - type: recall_at_1
      value: 29.875
    - type: recall_at_10
      value: 59.065999999999995
    - type: recall_at_100
      value: 80.923
    - type: recall_at_1000
      value: 93.927
    - type: recall_at_3
      value: 44.462
    - type: recall_at_5
      value: 51.89
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.94
    - type: map_at_10
      value: 35.125
    - type: map_at_100
      value: 36.476
    - type: map_at_1000
      value: 36.579
    - type: map_at_3
      value: 31.840000000000003
    - type: map_at_5
      value: 33.647
    - type: mrr_at_1
      value: 30.936000000000003
    - type: mrr_at_10
      value: 40.637
    - type: mrr_at_100
      value: 41.471000000000004
    - type: mrr_at_1000
      value: 41.525
    - type: mrr_at_3
      value: 38.013999999999996
    - type: mrr_at_5
      value: 39.469
    - type: ndcg_at_1
      value: 30.936000000000003
    - type: ndcg_at_10
      value: 41.295
    - type: ndcg_at_100
      value: 46.92
    - type: ndcg_at_1000
      value: 49.183
    - type: ndcg_at_3
      value: 35.811
    - type: ndcg_at_5
      value: 38.306000000000004
    - type: precision_at_1
      value: 30.936000000000003
    - type: precision_at_10
      value: 7.728
    - type: precision_at_100
      value: 1.226
    - type: precision_at_1000
      value: 0.158
    - type: precision_at_3
      value: 17.237
    - type: precision_at_5
      value: 12.42
    - type: recall_at_1
      value: 24.94
    - type: recall_at_10
      value: 54.235
    - type: recall_at_100
      value: 78.314
    - type: recall_at_1000
      value: 93.973
    - type: recall_at_3
      value: 38.925
    - type: recall_at_5
      value: 45.505
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 26.250833333333333
    - type: map_at_10
      value: 35.46875
    - type: map_at_100
      value: 36.667
    - type: map_at_1000
      value: 36.78025
    - type: map_at_3
      value: 32.56733333333334
    - type: map_at_5
      value: 34.20333333333333
    - type: mrr_at_1
      value: 30.8945
    - type: mrr_at_10
      value: 39.636833333333335
    - type: mrr_at_100
      value: 40.46508333333333
    - type: mrr_at_1000
      value: 40.521249999999995
    - type: mrr_at_3
      value: 37.140166666666666
    - type: mrr_at_5
      value: 38.60999999999999
    - type: ndcg_at_1
      value: 30.8945
    - type: ndcg_at_10
      value: 40.93441666666667
    - type: ndcg_at_100
      value: 46.062416666666664
    - type: ndcg_at_1000
      value: 48.28341666666667
    - type: ndcg_at_3
      value: 35.97575
    - type: ndcg_at_5
      value: 38.3785
    - type: precision_at_1
      value: 30.8945
    - type: precision_at_10
      value: 7.180250000000001
    - type: precision_at_100
      value: 1.1468333333333334
    - type: precision_at_1000
      value: 0.15283333333333332
    - type: precision_at_3
      value: 16.525583333333334
    - type: precision_at_5
      value: 11.798333333333332
    - type: recall_at_1
      value: 26.250833333333333
    - type: recall_at_10
      value: 52.96108333333333
    - type: recall_at_100
      value: 75.45908333333334
    - type: recall_at_1000
      value: 90.73924999999998
    - type: recall_at_3
      value: 39.25483333333333
    - type: recall_at_5
      value: 45.37950000000001
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.595
    - type: map_at_10
      value: 31.747999999999998
    - type: map_at_100
      value: 32.62
    - type: map_at_1000
      value: 32.713
    - type: map_at_3
      value: 29.48
    - type: map_at_5
      value: 30.635
    - type: mrr_at_1
      value: 27.607
    - type: mrr_at_10
      value: 34.449000000000005
    - type: mrr_at_100
      value: 35.182
    - type: mrr_at_1000
      value: 35.254000000000005
    - type: mrr_at_3
      value: 32.413
    - type: mrr_at_5
      value: 33.372
    - type: ndcg_at_1
      value: 27.607
    - type: ndcg_at_10
      value: 36.041000000000004
    - type: ndcg_at_100
      value: 40.514
    - type: ndcg_at_1000
      value: 42.851
    - type: ndcg_at_3
      value: 31.689
    - type: ndcg_at_5
      value: 33.479
    - type: precision_at_1
      value: 27.607
    - type: precision_at_10
      value: 5.66
    - type: precision_at_100
      value: 0.868
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 13.446
    - type: precision_at_5
      value: 9.264
    - type: recall_at_1
      value: 24.595
    - type: recall_at_10
      value: 46.79
    - type: recall_at_100
      value: 67.413
    - type: recall_at_1000
      value: 84.753
    - type: recall_at_3
      value: 34.644999999999996
    - type: recall_at_5
      value: 39.09
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 17.333000000000002
    - type: map_at_10
      value: 24.427
    - type: map_at_100
      value: 25.576
    - type: map_at_1000
      value: 25.692999999999998
    - type: map_at_3
      value: 22.002
    - type: map_at_5
      value: 23.249
    - type: mrr_at_1
      value: 20.716
    - type: mrr_at_10
      value: 28.072000000000003
    - type: mrr_at_100
      value: 29.067
    - type: mrr_at_1000
      value: 29.137
    - type: mrr_at_3
      value: 25.832
    - type: mrr_at_5
      value: 27.045
    - type: ndcg_at_1
      value: 20.716
    - type: ndcg_at_10
      value: 29.109
    - type: ndcg_at_100
      value: 34.797
    - type: ndcg_at_1000
      value: 37.503
    - type: ndcg_at_3
      value: 24.668
    - type: ndcg_at_5
      value: 26.552999999999997
    - type: precision_at_1
      value: 20.716
    - type: precision_at_10
      value: 5.351
    - type: precision_at_100
      value: 0.955
    - type: precision_at_1000
      value: 0.136
    - type: precision_at_3
      value: 11.584999999999999
    - type: precision_at_5
      value: 8.362
    - type: recall_at_1
      value: 17.333000000000002
    - type: recall_at_10
      value: 39.604
    - type: recall_at_100
      value: 65.525
    - type: recall_at_1000
      value: 84.651
    - type: recall_at_3
      value: 27.199
    - type: recall_at_5
      value: 32.019
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 26.342
    - type: map_at_10
      value: 35.349000000000004
    - type: map_at_100
      value: 36.443
    - type: map_at_1000
      value: 36.548
    - type: map_at_3
      value: 32.307
    - type: map_at_5
      value: 34.164
    - type: mrr_at_1
      value: 31.063000000000002
    - type: mrr_at_10
      value: 39.703
    - type: mrr_at_100
      value: 40.555
    - type: mrr_at_1000
      value: 40.614
    - type: mrr_at_3
      value: 37.141999999999996
    - type: mrr_at_5
      value: 38.812000000000005
    - type: ndcg_at_1
      value: 31.063000000000002
    - type: ndcg_at_10
      value: 40.873
    - type: ndcg_at_100
      value: 45.896
    - type: ndcg_at_1000
      value: 48.205999999999996
    - type: ndcg_at_3
      value: 35.522
    - type: ndcg_at_5
      value: 38.419
    - type: precision_at_1
      value: 31.063000000000002
    - type: precision_at_10
      value: 6.866
    - type: precision_at_100
      value: 1.053
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 16.014
    - type: precision_at_5
      value: 11.604000000000001
    - type: recall_at_1
      value: 26.342
    - type: recall_at_10
      value: 53.40200000000001
    - type: recall_at_100
      value: 75.251
    - type: recall_at_1000
      value: 91.13799999999999
    - type: recall_at_3
      value: 39.103
    - type: recall_at_5
      value: 46.357
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 23.71
    - type: map_at_10
      value: 32.153999999999996
    - type: map_at_100
      value: 33.821
    - type: map_at_1000
      value: 34.034
    - type: map_at_3
      value: 29.376
    - type: map_at_5
      value: 30.878
    - type: mrr_at_1
      value: 28.458
    - type: mrr_at_10
      value: 36.775999999999996
    - type: mrr_at_100
      value: 37.804
    - type: mrr_at_1000
      value: 37.858999999999995
    - type: mrr_at_3
      value: 34.123999999999995
    - type: mrr_at_5
      value: 35.596
    - type: ndcg_at_1
      value: 28.458
    - type: ndcg_at_10
      value: 37.858999999999995
    - type: ndcg_at_100
      value: 44.194
    - type: ndcg_at_1000
      value: 46.744
    - type: ndcg_at_3
      value: 33.348
    - type: ndcg_at_5
      value: 35.448
    - type: precision_at_1
      value: 28.458
    - type: precision_at_10
      value: 7.4510000000000005
    - type: precision_at_100
      value: 1.5
    - type: precision_at_1000
      value: 0.23700000000000002
    - type: precision_at_3
      value: 15.809999999999999
    - type: precision_at_5
      value: 11.462
    - type: recall_at_1
      value: 23.71
    - type: recall_at_10
      value: 48.272999999999996
    - type: recall_at_100
      value: 77.134
    - type: recall_at_1000
      value: 93.001
    - type: recall_at_3
      value: 35.480000000000004
    - type: recall_at_5
      value: 41.19
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 21.331
    - type: map_at_10
      value: 28.926000000000002
    - type: map_at_100
      value: 29.855999999999998
    - type: map_at_1000
      value: 29.957
    - type: map_at_3
      value: 26.395999999999997
    - type: map_at_5
      value: 27.933000000000003
    - type: mrr_at_1
      value: 23.105
    - type: mrr_at_10
      value: 31.008000000000003
    - type: mrr_at_100
      value: 31.819999999999997
    - type: mrr_at_1000
      value: 31.887999999999998
    - type: mrr_at_3
      value: 28.466
    - type: mrr_at_5
      value: 30.203000000000003
    - type: ndcg_at_1
      value: 23.105
    - type: ndcg_at_10
      value: 33.635999999999996
    - type: ndcg_at_100
      value: 38.277
    - type: ndcg_at_1000
      value: 40.907
    - type: ndcg_at_3
      value: 28.791
    - type: ndcg_at_5
      value: 31.528
    - type: precision_at_1
      value: 23.105
    - type: precision_at_10
      value: 5.323
    - type: precision_at_100
      value: 0.815
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 12.384
    - type: precision_at_5
      value: 9.02
    - type: recall_at_1
      value: 21.331
    - type: recall_at_10
      value: 46.018
    - type: recall_at_100
      value: 67.364
    - type: recall_at_1000
      value: 86.97
    - type: recall_at_3
      value: 33.395
    - type: recall_at_5
      value: 39.931
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 17.011000000000003
    - type: map_at_10
      value: 28.816999999999997
    - type: map_at_100
      value: 30.761
    - type: map_at_1000
      value: 30.958000000000002
    - type: map_at_3
      value: 24.044999999999998
    - type: map_at_5
      value: 26.557
    - type: mrr_at_1
      value: 38.696999999999996
    - type: mrr_at_10
      value: 50.464
    - type: mrr_at_100
      value: 51.193999999999996
    - type: mrr_at_1000
      value: 51.219
    - type: mrr_at_3
      value: 47.339999999999996
    - type: mrr_at_5
      value: 49.346000000000004
    - type: ndcg_at_1
      value: 38.696999999999996
    - type: ndcg_at_10
      value: 38.53
    - type: ndcg_at_100
      value: 45.525
    - type: ndcg_at_1000
      value: 48.685
    - type: ndcg_at_3
      value: 32.282
    - type: ndcg_at_5
      value: 34.482
    - type: precision_at_1
      value: 38.696999999999996
    - type: precision_at_10
      value: 11.895999999999999
    - type: precision_at_100
      value: 1.95
    - type: precision_at_1000
      value: 0.254
    - type: precision_at_3
      value: 24.038999999999998
    - type: precision_at_5
      value: 18.332
    - type: recall_at_1
      value: 17.011000000000003
    - type: recall_at_10
      value: 44.452999999999996
    - type: recall_at_100
      value: 68.223
    - type: recall_at_1000
      value: 85.653
    - type: recall_at_3
      value: 28.784
    - type: recall_at_5
      value: 35.66
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 9.516
    - type: map_at_10
      value: 21.439
    - type: map_at_100
      value: 31.517
    - type: map_at_1000
      value: 33.267
    - type: map_at_3
      value: 15.004999999999999
    - type: map_at_5
      value: 17.793999999999997
    - type: mrr_at_1
      value: 71.25
    - type: mrr_at_10
      value: 79.071
    - type: mrr_at_100
      value: 79.325
    - type: mrr_at_1000
      value: 79.33
    - type: mrr_at_3
      value: 77.708
    - type: mrr_at_5
      value: 78.546
    - type: ndcg_at_1
      value: 58.62500000000001
    - type: ndcg_at_10
      value: 44.889
    - type: ndcg_at_100
      value: 50.536
    - type: ndcg_at_1000
      value: 57.724
    - type: ndcg_at_3
      value: 49.32
    - type: ndcg_at_5
      value: 46.775
    - type: precision_at_1
      value: 71.25
    - type: precision_at_10
      value: 36.175000000000004
    - type: precision_at_100
      value: 11.940000000000001
    - type: precision_at_1000
      value: 2.178
    - type: precision_at_3
      value: 53.583000000000006
    - type: precision_at_5
      value: 45.550000000000004
    - type: recall_at_1
      value: 9.516
    - type: recall_at_10
      value: 27.028000000000002
    - type: recall_at_100
      value: 57.581
    - type: recall_at_1000
      value: 80.623
    - type: recall_at_3
      value: 16.313
    - type: recall_at_5
      value: 20.674
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 51.74999999999999
    - type: f1
      value: 46.46706502669774
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 77.266
    - type: map_at_10
      value: 84.89999999999999
    - type: map_at_100
      value: 85.109
    - type: map_at_1000
      value: 85.123
    - type: map_at_3
      value: 83.898
    - type: map_at_5
      value: 84.541
    - type: mrr_at_1
      value: 83.138
    - type: mrr_at_10
      value: 89.37
    - type: mrr_at_100
      value: 89.432
    - type: mrr_at_1000
      value: 89.43299999999999
    - type: mrr_at_3
      value: 88.836
    - type: mrr_at_5
      value: 89.21
    - type: ndcg_at_1
      value: 83.138
    - type: ndcg_at_10
      value: 88.244
    - type: ndcg_at_100
      value: 88.98700000000001
    - type: ndcg_at_1000
      value: 89.21900000000001
    - type: ndcg_at_3
      value: 86.825
    - type: ndcg_at_5
      value: 87.636
    - type: precision_at_1
      value: 83.138
    - type: precision_at_10
      value: 10.47
    - type: precision_at_100
      value: 1.1079999999999999
    - type: precision_at_1000
      value: 0.11499999999999999
    - type: precision_at_3
      value: 32.933
    - type: precision_at_5
      value: 20.36
    - type: recall_at_1
      value: 77.266
    - type: recall_at_10
      value: 94.063
    - type: recall_at_100
      value: 96.993
    - type: recall_at_1000
      value: 98.414
    - type: recall_at_3
      value: 90.228
    - type: recall_at_5
      value: 92.328
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 22.319
    - type: map_at_10
      value: 36.943
    - type: map_at_100
      value: 38.951
    - type: map_at_1000
      value: 39.114
    - type: map_at_3
      value: 32.82
    - type: map_at_5
      value: 34.945
    - type: mrr_at_1
      value: 44.135999999999996
    - type: mrr_at_10
      value: 53.071999999999996
    - type: mrr_at_100
      value: 53.87
    - type: mrr_at_1000
      value: 53.90200000000001
    - type: mrr_at_3
      value: 50.77199999999999
    - type: mrr_at_5
      value: 52.129999999999995
    - type: ndcg_at_1
      value: 44.135999999999996
    - type: ndcg_at_10
      value: 44.836
    - type: ndcg_at_100
      value: 51.754
    - type: ndcg_at_1000
      value: 54.36
    - type: ndcg_at_3
      value: 41.658
    - type: ndcg_at_5
      value: 42.354
    - type: precision_at_1
      value: 44.135999999999996
    - type: precision_at_10
      value: 12.284
    - type: precision_at_100
      value: 1.952
    - type: precision_at_1000
      value: 0.242
    - type: precision_at_3
      value: 27.828999999999997
    - type: precision_at_5
      value: 20.093
    - type: recall_at_1
      value: 22.319
    - type: recall_at_10
      value: 51.528
    - type: recall_at_100
      value: 76.70700000000001
    - type: recall_at_1000
      value: 92.143
    - type: recall_at_3
      value: 38.641
    - type: recall_at_5
      value: 43.653999999999996
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 40.182
    - type: map_at_10
      value: 65.146
    - type: map_at_100
      value: 66.023
    - type: map_at_1000
      value: 66.078
    - type: map_at_3
      value: 61.617999999999995
    - type: map_at_5
      value: 63.82299999999999
    - type: mrr_at_1
      value: 80.365
    - type: mrr_at_10
      value: 85.79
    - type: mrr_at_100
      value: 85.963
    - type: mrr_at_1000
      value: 85.968
    - type: mrr_at_3
      value: 84.952
    - type: mrr_at_5
      value: 85.503
    - type: ndcg_at_1
      value: 80.365
    - type: ndcg_at_10
      value: 73.13499999999999
    - type: ndcg_at_100
      value: 76.133
    - type: ndcg_at_1000
      value: 77.151
    - type: ndcg_at_3
      value: 68.255
    - type: ndcg_at_5
      value: 70.978
    - type: precision_at_1
      value: 80.365
    - type: precision_at_10
      value: 15.359
    - type: precision_at_100
      value: 1.7690000000000001
    - type: precision_at_1000
      value: 0.19
    - type: precision_at_3
      value: 44.024
    - type: precision_at_5
      value: 28.555999999999997
    - type: recall_at_1
      value: 40.182
    - type: recall_at_10
      value: 76.793
    - type: recall_at_100
      value: 88.474
    - type: recall_at_1000
      value: 95.159
    - type: recall_at_3
      value: 66.036
    - type: recall_at_5
      value: 71.391
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 92.7796
    - type: ap
      value: 89.24883716810874
    - type: f1
      value: 92.7706903433313
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 22.016
    - type: map_at_10
      value: 34.408
    - type: map_at_100
      value: 35.592
    - type: map_at_1000
      value: 35.64
    - type: map_at_3
      value: 30.459999999999997
    - type: map_at_5
      value: 32.721000000000004
    - type: mrr_at_1
      value: 22.593
    - type: mrr_at_10
      value: 34.993
    - type: mrr_at_100
      value: 36.113
    - type: mrr_at_1000
      value: 36.156
    - type: mrr_at_3
      value: 31.101
    - type: mrr_at_5
      value: 33.364
    - type: ndcg_at_1
      value: 22.579
    - type: ndcg_at_10
      value: 41.404999999999994
    - type: ndcg_at_100
      value: 47.018
    - type: ndcg_at_1000
      value: 48.211999999999996
    - type: ndcg_at_3
      value: 33.389
    - type: ndcg_at_5
      value: 37.425000000000004
    - type: precision_at_1
      value: 22.579
    - type: precision_at_10
      value: 6.59
    - type: precision_at_100
      value: 0.938
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.241000000000001
    - type: precision_at_5
      value: 10.59
    - type: recall_at_1
      value: 22.016
    - type: recall_at_10
      value: 62.927
    - type: recall_at_100
      value: 88.72
    - type: recall_at_1000
      value: 97.80799999999999
    - type: recall_at_3
      value: 41.229
    - type: recall_at_5
      value: 50.88
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 94.01732786137711
    - type: f1
      value: 93.76353126402202
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 76.91746466028272
    - type: f1
      value: 57.715651682646765
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 76.5030262273033
    - type: f1
      value: 74.6693629986121
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 79.74781439139207
    - type: f1
      value: 79.96684171018774
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 33.2156206892017
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 31.180539484816137
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 32.51125957874274
    - type: mrr
      value: 33.777037359249995
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 7.248
    - type: map_at_10
      value: 15.340000000000002
    - type: map_at_100
      value: 19.591
    - type: map_at_1000
      value: 21.187
    - type: map_at_3
      value: 11.329
    - type: map_at_5
      value: 13.209999999999999
    - type: mrr_at_1
      value: 47.678
    - type: mrr_at_10
      value: 57.493
    - type: mrr_at_100
      value: 58.038999999999994
    - type: mrr_at_1000
      value: 58.07
    - type: mrr_at_3
      value: 55.36600000000001
    - type: mrr_at_5
      value: 56.635999999999996
    - type: ndcg_at_1
      value: 46.129999999999995
    - type: ndcg_at_10
      value: 38.653999999999996
    - type: ndcg_at_100
      value: 36.288
    - type: ndcg_at_1000
      value: 44.765
    - type: ndcg_at_3
      value: 43.553
    - type: ndcg_at_5
      value: 41.317
    - type: precision_at_1
      value: 47.368
    - type: precision_at_10
      value: 28.669
    - type: precision_at_100
      value: 9.158
    - type: precision_at_1000
      value: 2.207
    - type: precision_at_3
      value: 40.97
    - type: precision_at_5
      value: 35.604
    - type: recall_at_1
      value: 7.248
    - type: recall_at_10
      value: 19.46
    - type: recall_at_100
      value: 37.214000000000006
    - type: recall_at_1000
      value: 67.64099999999999
    - type: recall_at_3
      value: 12.025
    - type: recall_at_5
      value: 15.443999999999999
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 31.595000000000002
    - type: map_at_10
      value: 47.815999999999995
    - type: map_at_100
      value: 48.811
    - type: map_at_1000
      value: 48.835
    - type: map_at_3
      value: 43.225
    - type: map_at_5
      value: 46.017
    - type: mrr_at_1
      value: 35.689
    - type: mrr_at_10
      value: 50.341
    - type: mrr_at_100
      value: 51.044999999999995
    - type: mrr_at_1000
      value: 51.062
    - type: mrr_at_3
      value: 46.553
    - type: mrr_at_5
      value: 48.918
    - type: ndcg_at_1
      value: 35.66
    - type: ndcg_at_10
      value: 55.859
    - type: ndcg_at_100
      value: 59.864
    - type: ndcg_at_1000
      value: 60.419999999999995
    - type: ndcg_at_3
      value: 47.371
    - type: ndcg_at_5
      value: 51.995000000000005
    - type: precision_at_1
      value: 35.66
    - type: precision_at_10
      value: 9.27
    - type: precision_at_100
      value: 1.1520000000000001
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 21.63
    - type: precision_at_5
      value: 15.655
    - type: recall_at_1
      value: 31.595000000000002
    - type: recall_at_10
      value: 77.704
    - type: recall_at_100
      value: 94.774
    - type: recall_at_1000
      value: 98.919
    - type: recall_at_3
      value: 56.052
    - type: recall_at_5
      value: 66.623
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 71.489
    - type: map_at_10
      value: 85.411
    - type: map_at_100
      value: 86.048
    - type: map_at_1000
      value: 86.064
    - type: map_at_3
      value: 82.587
    - type: map_at_5
      value: 84.339
    - type: mrr_at_1
      value: 82.28
    - type: mrr_at_10
      value: 88.27199999999999
    - type: mrr_at_100
      value: 88.362
    - type: mrr_at_1000
      value: 88.362
    - type: mrr_at_3
      value: 87.372
    - type: mrr_at_5
      value: 87.995
    - type: ndcg_at_1
      value: 82.27
    - type: ndcg_at_10
      value: 89.023
    - type: ndcg_at_100
      value: 90.191
    - type: ndcg_at_1000
      value: 90.266
    - type: ndcg_at_3
      value: 86.37
    - type: ndcg_at_5
      value: 87.804
    - type: precision_at_1
      value: 82.27
    - type: precision_at_10
      value: 13.469000000000001
    - type: precision_at_100
      value: 1.533
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.797
    - type: precision_at_5
      value: 24.734
    - type: recall_at_1
      value: 71.489
    - type: recall_at_10
      value: 95.824
    - type: recall_at_100
      value: 99.70599999999999
    - type: recall_at_1000
      value: 99.979
    - type: recall_at_3
      value: 88.099
    - type: recall_at_5
      value: 92.285
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 60.52398807444541
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 65.34855891507871
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.188000000000001
    - type: map_at_10
      value: 13.987
    - type: map_at_100
      value: 16.438
    - type: map_at_1000
      value: 16.829
    - type: map_at_3
      value: 9.767000000000001
    - type: map_at_5
      value: 11.912
    - type: mrr_at_1
      value: 25.6
    - type: mrr_at_10
      value: 37.744
    - type: mrr_at_100
      value: 38.847
    - type: mrr_at_1000
      value: 38.894
    - type: mrr_at_3
      value: 34.166999999999994
    - type: mrr_at_5
      value: 36.207
    - type: ndcg_at_1
      value: 25.6
    - type: ndcg_at_10
      value: 22.980999999999998
    - type: ndcg_at_100
      value: 32.039
    - type: ndcg_at_1000
      value: 38.157000000000004
    - type: ndcg_at_3
      value: 21.567
    - type: ndcg_at_5
      value: 19.070999999999998
    - type: precision_at_1
      value: 25.6
    - type: precision_at_10
      value: 12.02
    - type: precision_at_100
      value: 2.5100000000000002
    - type: precision_at_1000
      value: 0.396
    - type: precision_at_3
      value: 20.333000000000002
    - type: precision_at_5
      value: 16.98
    - type: recall_at_1
      value: 5.188000000000001
    - type: recall_at_10
      value: 24.372
    - type: recall_at_100
      value: 50.934999999999995
    - type: recall_at_1000
      value: 80.477
    - type: recall_at_3
      value: 12.363
    - type: recall_at_5
      value: 17.203
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_pearson
      value: 87.24286275535398
    - type: cos_sim_spearman
      value: 82.62333770991818
    - type: euclidean_pearson
      value: 84.60353717637284
    - type: euclidean_spearman
      value: 82.32990108810047
    - type: manhattan_pearson
      value: 84.6089049738196
    - type: manhattan_spearman
      value: 82.33361785438936
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 87.87428858503165
    - type: cos_sim_spearman
      value: 79.09145886519929
    - type: euclidean_pearson
      value: 86.42669231664036
    - type: euclidean_spearman
      value: 80.03127375435449
    - type: manhattan_pearson
      value: 86.41330338305022
    - type: manhattan_spearman
      value: 80.02492538673368
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 88.67912277322645
    - type: cos_sim_spearman
      value: 89.6171319711762
    - type: euclidean_pearson
      value: 86.56571917398725
    - type: euclidean_spearman
      value: 87.71216907898948
    - type: manhattan_pearson
      value: 86.57459050182473
    - type: manhattan_spearman
      value: 87.71916648349993
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 86.71957379085862
    - type: cos_sim_spearman
      value: 85.01784075851465
    - type: euclidean_pearson
      value: 84.7407848472801
    - type: euclidean_spearman
      value: 84.61063091345538
    - type: manhattan_pearson
      value: 84.71494352494403
    - type: manhattan_spearman
      value: 84.58772077604254
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 88.40508326325175
    - type: cos_sim_spearman
      value: 89.50912897763186
    - type: euclidean_pearson
      value: 87.82349070086627
    - type: euclidean_spearman
      value: 88.44179162727521
    - type: manhattan_pearson
      value: 87.80181927025595
    - type: manhattan_spearman
      value: 88.43205129636243
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 85.35846741715478
    - type: cos_sim_spearman
      value: 86.61172476741842
    - type: euclidean_pearson
      value: 84.60123125491637
    - type: euclidean_spearman
      value: 85.3001948141827
    - type: manhattan_pearson
      value: 84.56231142658329
    - type: manhattan_spearman
      value: 85.23579900798813
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 88.94539129818824
    - type: cos_sim_spearman
      value: 88.99349064256742
    - type: euclidean_pearson
      value: 88.7142444640351
    - type: euclidean_spearman
      value: 88.34120813505011
    - type: manhattan_pearson
      value: 88.70363008238084
    - type: manhattan_spearman
      value: 88.31952816956954
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 68.29910260369893
    - type: cos_sim_spearman
      value: 68.79263346213466
    - type: euclidean_pearson
      value: 68.41627521422252
    - type: euclidean_spearman
      value: 66.61602587398579
    - type: manhattan_pearson
      value: 68.49402183447361
    - type: manhattan_spearman
      value: 66.80157792354453
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 87.43703906343708
    - type: cos_sim_spearman
      value: 89.06081805093662
    - type: euclidean_pearson
      value: 87.48311456299662
    - type: euclidean_spearman
      value: 88.07417597580013
    - type: manhattan_pearson
      value: 87.48202249768894
    - type: manhattan_spearman
      value: 88.04758031111642
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 87.49080620485203
    - type: mrr
      value: 96.19145378949301
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 59.317
    - type: map_at_10
      value: 69.296
    - type: map_at_100
      value: 69.738
    - type: map_at_1000
      value: 69.759
    - type: map_at_3
      value: 66.12599999999999
    - type: map_at_5
      value: 67.532
    - type: mrr_at_1
      value: 62
    - type: mrr_at_10
      value: 70.176
    - type: mrr_at_100
      value: 70.565
    - type: mrr_at_1000
      value: 70.583
    - type: mrr_at_3
      value: 67.833
    - type: mrr_at_5
      value: 68.93299999999999
    - type: ndcg_at_1
      value: 62
    - type: ndcg_at_10
      value: 74.069
    - type: ndcg_at_100
      value: 76.037
    - type: ndcg_at_1000
      value: 76.467
    - type: ndcg_at_3
      value: 68.628
    - type: ndcg_at_5
      value: 70.57600000000001
    - type: precision_at_1
      value: 62
    - type: precision_at_10
      value: 10
    - type: precision_at_100
      value: 1.097
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 26.667
    - type: precision_at_5
      value: 17.4
    - type: recall_at_1
      value: 59.317
    - type: recall_at_10
      value: 87.822
    - type: recall_at_100
      value: 96.833
    - type: recall_at_1000
      value: 100
    - type: recall_at_3
      value: 73.06099999999999
    - type: recall_at_5
      value: 77.928
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.88910891089108
    - type: cos_sim_ap
      value: 97.236958456951
    - type: cos_sim_f1
      value: 94.39999999999999
    - type: cos_sim_precision
      value: 94.39999999999999
    - type: cos_sim_recall
      value: 94.39999999999999
    - type: dot_accuracy
      value: 99.82574257425742
    - type: dot_ap
      value: 94.94344759441888
    - type: dot_f1
      value: 91.17352056168507
    - type: dot_precision
      value: 91.44869215291752
    - type: dot_recall
      value: 90.9
    - type: euclidean_accuracy
      value: 99.88415841584158
    - type: euclidean_ap
      value: 97.2044250782305
    - type: euclidean_f1
      value: 94.210786739238
    - type: euclidean_precision
      value: 93.24191968658178
    - type: euclidean_recall
      value: 95.19999999999999
    - type: manhattan_accuracy
      value: 99.88613861386139
    - type: manhattan_ap
      value: 97.20683205497689
    - type: manhattan_f1
      value: 94.2643391521197
    - type: manhattan_precision
      value: 94.02985074626866
    - type: manhattan_recall
      value: 94.5
    - type: max_accuracy
      value: 99.88910891089108
    - type: max_ap
      value: 97.236958456951
    - type: max_f1
      value: 94.39999999999999
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 66.53940781726187
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 36.71865011295108
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 55.3218674533331
    - type: mrr
      value: 56.28279910449028
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.723915667479673
    - type: cos_sim_spearman
      value: 32.029070449745234
    - type: dot_pearson
      value: 28.864944212481454
    - type: dot_spearman
      value: 27.939266999596725
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.231
    - type: map_at_10
      value: 1.949
    - type: map_at_100
      value: 10.023
    - type: map_at_1000
      value: 23.485
    - type: map_at_3
      value: 0.652
    - type: map_at_5
      value: 1.054
    - type: mrr_at_1
      value: 86
    - type: mrr_at_10
      value: 92.067
    - type: mrr_at_100
      value: 92.067
    - type: mrr_at_1000
      value: 92.067
    - type: mrr_at_3
      value: 91.667
    - type: mrr_at_5
      value: 92.067
    - type: ndcg_at_1
      value: 83
    - type: ndcg_at_10
      value: 76.32900000000001
    - type: ndcg_at_100
      value: 54.662
    - type: ndcg_at_1000
      value: 48.062
    - type: ndcg_at_3
      value: 81.827
    - type: ndcg_at_5
      value: 80.664
    - type: precision_at_1
      value: 86
    - type: precision_at_10
      value: 80
    - type: precision_at_100
      value: 55.48
    - type: precision_at_1000
      value: 20.938000000000002
    - type: precision_at_3
      value: 85.333
    - type: precision_at_5
      value: 84.39999999999999
    - type: recall_at_1
      value: 0.231
    - type: recall_at_10
      value: 2.158
    - type: recall_at_100
      value: 13.344000000000001
    - type: recall_at_1000
      value: 44.31
    - type: recall_at_3
      value: 0.6779999999999999
    - type: recall_at_5
      value: 1.13
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 2.524
    - type: map_at_10
      value: 10.183
    - type: map_at_100
      value: 16.625
    - type: map_at_1000
      value: 18.017
    - type: map_at_3
      value: 5.169
    - type: map_at_5
      value: 6.772
    - type: mrr_at_1
      value: 32.653
    - type: mrr_at_10
      value: 47.128
    - type: mrr_at_100
      value: 48.458
    - type: mrr_at_1000
      value: 48.473
    - type: mrr_at_3
      value: 44.897999999999996
    - type: mrr_at_5
      value: 45.306000000000004
    - type: ndcg_at_1
      value: 30.612000000000002
    - type: ndcg_at_10
      value: 24.928
    - type: ndcg_at_100
      value: 37.613
    - type: ndcg_at_1000
      value: 48.528
    - type: ndcg_at_3
      value: 28.829
    - type: ndcg_at_5
      value: 25.237
    - type: precision_at_1
      value: 32.653
    - type: precision_at_10
      value: 22.448999999999998
    - type: precision_at_100
      value: 8.02
    - type: precision_at_1000
      value: 1.537
    - type: precision_at_3
      value: 30.612000000000002
    - type: precision_at_5
      value: 24.490000000000002
    - type: recall_at_1
      value: 2.524
    - type: recall_at_10
      value: 16.38
    - type: recall_at_100
      value: 49.529
    - type: recall_at_1000
      value: 83.598
    - type: recall_at_3
      value: 6.411
    - type: recall_at_5
      value: 8.932
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 71.09020000000001
    - type: ap
      value: 14.451710060978993
    - type: f1
      value: 54.7874410609049
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 59.745331069609506
    - type: f1
      value: 60.08387848592697
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 51.71549485462037
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 87.39345532574357
    - type: cos_sim_ap
      value: 78.16796549696478
    - type: cos_sim_f1
      value: 71.27713276123171
    - type: cos_sim_precision
      value: 68.3115626511853
    - type: cos_sim_recall
      value: 74.51187335092348
    - type: dot_accuracy
      value: 85.12248912201228
    - type: dot_ap
      value: 69.26039256107077
    - type: dot_f1
      value: 65.04294321240867
    - type: dot_precision
      value: 63.251059586138126
    - type: dot_recall
      value: 66.93931398416886
    - type: euclidean_accuracy
      value: 87.07754664123503
    - type: euclidean_ap
      value: 77.7872176038945
    - type: euclidean_f1
      value: 70.85587801278899
    - type: euclidean_precision
      value: 66.3519115614924
    - type: euclidean_recall
      value: 76.01583113456465
    - type: manhattan_accuracy
      value: 87.07754664123503
    - type: manhattan_ap
      value: 77.7341400185556
    - type: manhattan_f1
      value: 70.80310880829015
    - type: manhattan_precision
      value: 69.54198473282443
    - type: manhattan_recall
      value: 72.1108179419525
    - type: max_accuracy
      value: 87.39345532574357
    - type: max_ap
      value: 78.16796549696478
    - type: max_f1
      value: 71.27713276123171
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 89.09457833663213
    - type: cos_sim_ap
      value: 86.33024314706873
    - type: cos_sim_f1
      value: 78.59623733719248
    - type: cos_sim_precision
      value: 74.13322413322413
    - type: cos_sim_recall
      value: 83.63104404065291
    - type: dot_accuracy
      value: 88.3086894089339
    - type: dot_ap
      value: 83.92225241805097
    - type: dot_f1
      value: 76.8721826377781
    - type: dot_precision
      value: 72.8168044077135
    - type: dot_recall
      value: 81.40591315060055
    - type: euclidean_accuracy
      value: 88.77052043311213
    - type: euclidean_ap
      value: 85.7410710218755
    - type: euclidean_f1
      value: 77.97705489398781
    - type: euclidean_precision
      value: 73.77713657598241
    - type: euclidean_recall
      value: 82.68401601478288
    - type: manhattan_accuracy
      value: 88.73753250281368
    - type: manhattan_ap
      value: 85.72867199072802
    - type: manhattan_f1
      value: 77.89774182922812
    - type: manhattan_precision
      value: 74.23787931635857
    - type: manhattan_recall
      value: 81.93717277486911
    - type: max_accuracy
      value: 89.09457833663213
    - type: max_ap
      value: 86.33024314706873
    - type: max_f1
      value: 78.59623733719248
license: mit
language:
- en
---


# [Universal AnglE Embedding](https://github.com/SeanLee97/AnglE)

📢 `WhereIsAI/UAE-Large-V1` **is licensed under MIT. Feel free to use it in any scenario.** 
**If you use it for academic papers, you could cite us via 👉 [citation info](#citation).**

**🤝 Follow us on:**

- GitHub: https://github.com/SeanLee97/AnglE.
- Preprint Paper: [AnglE-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)
- Conference Paper: [AoE: Angle-optimized Embeddings for Semantic Textual Similarity](https://aclanthology.org/2024.acl-long.101/) (ACL24)
- **📘 Documentation**: https://angle.readthedocs.io/en/latest/index.html

Welcome to using AnglE to train and infer powerful sentence embeddings.

**🏆 Achievements**

- 📅 May 16, 2024 | AnglE's paper is accepted by ACL 2024 Main Conference
- 📅 Dec 4, 2023 | 🔥 Our universal English sentence embedding `WhereIsAI/UAE-Large-V1` achieves **SOTA** on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of 64.64!

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/635cc29de7aef2358a9b03ee/jY3tr0DCMdyJXOihSqJFr.jpeg)


**🧑‍🤝‍🧑 Siblings:**

- [WhereIsAI/UAE-Code-Large-V1](https://huggingface.co/WhereIsAI/UAE-Code-Large-V1): This model can be used for code or GitHub issue similarity measurement.


# Usage
## 1. angle_emb

```bash
python -m pip install -U angle-emb
```

1) Non-Retrieval Tasks

There is no need to specify any prompts.

```python
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], normalize_embedding=True)

for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

2) Retrieval Tasks

For retrieval purposes, please use the prompt `Prompts.C` for query (not for document).

```python
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
qv = angle.encode(Prompts.C.format(text='what is the weather?'))
doc_vecs = angle.encode([
    'The weather is great!',
    'it is rainy today.',
    'i am going to bed'
])

for dv in doc_vecs:
    print(cosine_similarity(qv[0], dv))
```

## 2. sentence transformer


```python
from angle_emb import Prompts
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("WhereIsAI/UAE-Large-V1").cuda()

qv = model.encode(Prompts.C.format(text='what is the weather?'))
doc_vecs = model.encode([
    'The weather is great!',
    'it is rainy today.',
    'i am going to bed'
])

for dv in doc_vecs:
    print(1 - spatial.distance.cosine(qv, dv))
```


## 3. Infinity

[Infinity](https://github.com/michaelfeil/infinity) is a MIT licensed server for OpenAI-compatible deployment.
```
docker run --gpus all -v $PWD/data:/app/.cache -p "7997":"7997" \
michaelf34/infinity:latest \
v2 --model-id WhereIsAI/UAE-Large-V1 --revision "369c368f70f16a613f19f5598d4f12d9f44235d4" --dtype float16 --batch-size 32 --device cuda --engine torch --port 7997
```

# Citation 

If you use our pre-trained models, welcome to support us by citing our work:

```
@article{li2023angle,
  title={AnglE-optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```