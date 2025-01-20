<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-fairness/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  RecSys-Fairness: Development of an Fairness Strategy for Recommendation Systems
</h3>

<p align="center">Fairness algorithm aimed at reducing group unfairness in recommendation systems. </p>

## :page_with_curl: About the project <a name="-about"/></a>

In this study, we address the importance of promoting fairness in recommendation systems, which are highly susceptible to biases that can lead to unfair outcomes for different user groups. We developed a fairness algorithm aimed at mitigating these injustices, which was applied to the **MovieLens** dataset and analyzed based on the recommendations produced by the **ALS (Alternating Least Squares)** and **NCF (Neural Collaborative Filtering)** methods.

Users were grouped by activity level, gender, and age, and the results demonstrated the effectiveness of the fairness algorithm in substantially reducing group unfairness ($R_{grp}$) across all tested configurations, without causing significant losses in recommendation accuracy, measured by the **Root Mean Squared Error (RMSE)**.

In particular, a reduction in group unfairness of up to **65.57%** was observed in the ALS method. Additionally, we identified an optimal convergence of the fairness algorithm for an estimated number of matrices ($h$) between **10 and 15**, suggesting an effective balance point between promoting fairness and maintaining precision in recommendations.

In comparison with the available benchmarks, under identical experimental conditions, we managed to improve group unfairness reductions by approximately **6%** (from **59.77%** to **65.57%**).


### :notebook_with_decorative_cover: Algorithm <a name="-algorithm"/></a>

<img src="https://github.com/ravarmes/recsys-fairness/blob/main/assets/recsys-fairness-1.png" width="700">


### Files

| File                                 | Description                                                                                                                                                                                                                                   |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Class to promote fairness in recommendations of recommendation system algorithms.                                                                                                                                                              |
| AlgorithmUserFairness                | Classes to measure fairness (polarization, individual fairness, and group fairness) of recommendations of recommendation system algorithms.                                                                                                    |
| RecSys                               | A factory pattern class to instantiate a recommendation system based on string parameters.                                                                                                                                                    |
| RecSysALS                            | Alternating Least Squares (ALS) for Collaborative Filtering is an algorithm that iteratively optimizes two matrices to better predict user ratings on items, based on the idea of matrix factorization.                                         |
| RecSysNCF                            | Neural Collaborative Filtering (NCF): uses neural networks to model interactions between users and items.                                         |
| TestAlgorithmImpartiality_Age        | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by age (Age).                                                                                                                                      |
| TestAlgorithmImpartiality_Age_SaveTXT| Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by age (Age) saving the results in a TXT file.                                                                                                   |
| TestAlgorithmImpartiality_Gender     | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by gender (Gender).                                                                                                                               |
| TestAlgorithmImpartiality_Gender_SaveTXT | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by gender (Gender) saving the results in a TXT file.                                                                                         |
| TestAlgorithmImpartiality_NR         | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by number of ratings (NR).                                                                                                                        |
| TestAlgorithmImpartiality_NR_SaveTXT | Test script for the impartiality algorithm (AlgorithmImpartiality) considering user grouping by number of ratings (NR) saving the results in a TXT file.                                                                                       |
