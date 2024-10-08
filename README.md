<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-fairness/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  Developing a fairness strategy for recommendation systems
</h3>

<p align="center">Fairness algorithm aimed at reducing group unfairness in recommendation systems. </p>

## :page_with_curl: About the project <a name="-about"/></a>

In this study, we address the importance of promoting fairness in recommender systems, which are highly susceptible to biases that can result in unfair outcomes for different user groups. We developed a fairness algorithm aimed at mitigating these unfairnesses, categorizing users as favored or disfavored. The algorithm was applied to three existing datasets (MovieLens, Songs, and GoodBooks) and analyzed based on the recommendations produced by two methods: ALS (Alternating Least Squares) and KNN (K-Nearest Neighbors).

The results demonstrated the effectiveness of the fairness algorithm in substantially reducing group unfairness (\(R_{grp}\)) in all tested configurations, without causing significant losses in the accuracy of the recommendations, measured by the Root Mean Squared Error (\(RMSE\)). In particular, a reduction in group unfairness of up to 93.87% was observed in the Songs dataset. Additionally, we identified an optimal convergence of the fairness algorithm for a number of estimated matrices (\(h\)) between 5 and 10, suggesting an effective balance point between promoting fairness and maintaining precision in the recommendations.


### :balance_scale: Fairness Measures <a name="-measures"/></a>

* Individual fairness: For each user \(i\), we define \(ℓ_i\), the loss of user \(i\), as the mean squared error of the estimate over the known ratings of user \(i\):

* Group fairness: Let \(I\) be the set of all users/items and \(G = \{G_1, ..., G_{g}\}\) a partition of users/items into \(g\) groups, i.e., \(I = \cup_{i \in \{1, ..., g\}} G_i\). We define the loss of group \(i\) as the mean squared error of the estimate over all known ratings in group \(i\):


### :notebook_with_decorative_cover: Algorithm <a name="-algorithm"/></a>

<img src="https://github.com/ravarmes/recsys-fairness/blob/main/assets/recsys-fairness-1.png" width="700">


### Files

| File                                 | Description                                                                                                                                                                                                                                   |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Class to promote fairness in recommendations of recommendation system algorithms.                                                                                                                                                              |
| AlgorithmUserFairness                | Classes to measure fairness (polarization, individual fairness, and group fairness) of recommendations of recommendation system algorithms.                                                                                                    |
| RecSys                               | A factory pattern class to instantiate a recommendation system based on string parameters.                                                                                                                                                    |
| RecSysALS                            | Alternating Least Squares (ALS) for Collaborative Filtering is an algorithm that iteratively optimizes two matrices to better predict user ratings on items, based on the idea of matrix factorization.                                         |
| TestAlgorithmImpartiality_Age        | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by age (Age).                                                                                                                                      |
| TestAlgorithmImpartiality_Age_SaveTXT| Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by age (Age) saving the results in a TXT file.                                                                                                   |
| TestAlgorithmImpartiality_Gender     | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by gender (Gender).                                                                                                                               |
| TestAlgorithmImpartiality_Gender_SaveTXT | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by gender (Gender) saving the results in a TXT file.                                                                                         |
| TestAlgorithmImpartiality_NR         | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by number of ratings (NR).                                                                                                                        |
| TestAlgorithmImpartiality_NR_SaveTXT | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by number of ratings (NR) saving the results in a TXT file.                                                                                       |
