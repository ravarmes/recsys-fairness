from RecSys import RecSys
from AlgorithmUserFairness import Polarization, IndividualLossVariance, GroupLossVariance, RMSE
from AlgorithmImpartiality import AlgorithmImpartiality

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def print_results(dataset, algorithm, h, i, n_users, n_items, Rgrp, result, original=False):

    if original == True:
        h = 0

    print(f'Dataset: {dataset}; Algorithm: {algorithm}; n_users: {n_users}; n_items: {n_items}; h: {h}; i: {i}')
    print(f'Group Loss Variance (Rgrp): {Rgrp:.9f}')
    print(f'Root Mean Squared Error (RMSE): {result:.9f}')

    # Write the results to the file _results.txt
    with open('_results.txt', 'a') as file:
        file.write(f'\nDataset: {dataset}; Algorithm: {algorithm}; n_users: {n_users}; n_items: {n_items}; h: {h}; i: {i}\n')
        file.write(f'Group Loss Variance (Rgrp): {Rgrp:.9f}'.replace(".", ",") + '\n')
        file.write(f'Root Mean Squared Error (RMSE): {result:.9f}'.replace(".", ",") + '\n')

    experiment_results.append({
        'Dataset': dataset, 
        'Algorithm': algorithm, 
        'n_users': n_users, 
        'n_items': n_items, 
        'h': h, 
        'i': i, 
        'Rgrp': f"{Rgrp:.9f}".replace(".", ","),
        'RMSE': f"{result:.9f}".replace(".", ",")
    })


def split_train_test(X, test_size=0.2):
    X_train = pd.DataFrame(index=X.index, columns=X.columns, dtype=np.float64)
    X_test = pd.DataFrame(index=X.index, columns=X.columns, dtype=np.float64)

    for user in X.index:
        user_ratings = X.loc[user].dropna()
        # train_ratings, test_ratings = train_test_split(user_ratings, test_size=test_size, random_state=42)
        train_ratings, test_ratings = train_test_split(user_ratings, test_size=test_size)
        
        X_train.loc[user, train_ratings.index] = train_ratings
        X_test.loc[user, test_ratings.index] = test_ratings

    return X_train, X_test


experiment_results = []

# user and item filtering
# n_users=  300
# n_items= 1000
# n_users=  4
# n_items= 6
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use MovieLens with more ratings; False: otherwise

# dataset
# dataset = 'MovieLens-1M'    # reading data from 3883 movies and 6040 users
# dataset = 'Goodbooks-10k' # reading data from 10000 GoodBooks and 53424 users
# dataset = 'Songs'         # reading data from 19993 songs and 16000 users
# datasets = ['MovieLens-1M', 'Goodbooks-10k', 'Songs']
datasets = ['MovieLens-1M']

# recommendation algorithm
# algorithm = 'RecSysALS' # Alternating Least Squares (ALS) for Collaborative Filtering
# algorithm = 'RecSysKNN' # K-Nearest Neighbors for Recommender Systems
# algorithm = 'RecSysNMF' # Non-Negative Matrix Factorization for Recommender Systems
# algorithm = 'RecSysSGD' # Stochastic Gradient Descent for Recommender Systems
# algorithm = 'RecSysSVD' # Singular Value Decomposition for Recommender Systems
# algorithm = 'RecSysNCF' # Neural Collaborative Filtering
algorithms = ['RecSysALS', 'RecSysNCF']
# algorithms = ['RecSysNCF']

# estimated number of matrices (h)
# h = 3
# h = 5
# h = 10
# h = 15
# h = 20
# hs = [3, 5, 10, 15, 20, 23]
# hs = [15, 20, 23]
# hs = [23]
hs = [3, 5, 10]


iteration = [1, 2, 3]


for dataset in datasets:

    Data_path = "Data/"+ dataset    

    recsys = RecSys(n_users, n_items, top_users, top_items)

    X, users_info, items_info = recsys.read_dataset(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_items columns

    X_train, X_test = split_train_test(X)
    omega_train = ~X_train.isnull() # matrix X_train with True in cells with evaluations and False in cells not rated

    for algorithm in algorithms:

        X_est_train = recsys.compute_X_est(X_train, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

        print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS [before the impartiality algorithm] ------------")

        # Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
        ilv = IndividualLossVariance(X_train, omega_train, 1) #axis = 1 (0 rows e 1 columns)
        losses = ilv.get_losses(X_est_train)

        # Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
        # The loss of group i as the mean squared estimation error over all known ratings in group i
        list_users = X_est_train.index.tolist()

        masculine = users_info[users_info['Gender'] == 1].index.intersection(list_users).tolist()
        feminine = users_info[users_info['Gender'] == 2].index.intersection(list_users).tolist()

        G = {1: masculine, 2: feminine}
        G_index = {1: [183, 200, 272, 247, 220, 212, 251, 141, 175, 137, 99, 299, 94, 230, 87, 86, 7, 270, 60, 120, 48, 170, 205, 16, 30, 160, 41, 37, 96, 261, 69, 129, 176, 107, 19, 266, 123, 32, 13, 84, 271, 134, 64, 124, 288, 1, 227, 257, 279, 159, 285, 74, 276, 18, 248, 135, 67, 252, 80, 117, 194, 26, 17, 127, 56, 61, 138, 102, 295, 76, 126, 3, 77, 296, 207, 177, 292, 156, 161, 108, 27, 165, 191, 51, 4, 174, 5, 269, 168, 283, 70, 21, 209, 162, 289, 232, 198, 82, 66, 249, 273, 54, 238, 281, 72, 297, 184, 293, 105, 149, 49, 203, 188, 258, 10, 196, 210, 71, 139, 291, 226, 33, 233, 52, 142, 186, 118, 166, 189, 23, 95, 112, 50, 151, 181, 45, 88, 12, 89, 231, 55, 68, 147, 294, 169, 125, 208, 103, 93, 85, 28, 286, 259, 39, 58, 222, 267, 211, 223, 201, 152, 245, 224, 178, 143, 140, 229, 47, 154, 110, 277, 136, 0, 206, 40, 100, 250, 2, 65, 11, 278, 29, 246, 36, 83, 15, 22, 20, 24, 234, 287, 146, 256, 130, 280, 38, 131, 187, 218, 57, 164, 128, 78, 282, 172, 43, 44, 182, 132, 115, 298, 219, 75, 109, 265, 8, 90, 260, 239, 148, 202, 153, 262, 53, 79, 59, 263, 255, 284, 204, 9, 237, 6, 111, 268, 240, 274, 46, 275, 62], 2: [180, 221, 92, 145, 199, 34, 73, 14, 35, 213, 133, 225, 81, 63, 101, 155, 113, 214, 97, 195, 236, 215, 171, 91, 121, 185, 98, 116, 31, 114, 216, 157, 217, 167, 254, 42, 190, 197, 241, 243, 228, 104, 242, 264, 163, 144, 244, 235, 179, 122, 173, 150, 290, 158, 25, 253, 192, 119, 193, 106]}


        ##############################################################################################################################
        for h in hs:

            print("\n------------------------------------------------------------------")
            print(f"h: {h}")

            for i in iteration:

                algorithmImpartiality = AlgorithmImpartiality(X_train, omega_train, 1)
                list_X_est = algorithmImpartiality.evaluate(X_est_train, h) # calculates a list of h estimated matrices

                list_losses = []
                for X_est in list_X_est:
                    losses = ilv.get_losses(X_est)
                    list_losses.append(losses)

                Z = AlgorithmImpartiality.losses_to_Z(list_losses, n_users)
                list_Zs = AlgorithmImpartiality.matrices_Zs(Z, G_index)
                X_gurobi = AlgorithmImpartiality.make_matrix_X_gurobi(list_X_est, G, list_Zs) # calculate the recommendation matrix optimized by gurobi

                # --------------------------------------------------------------------------
                # Train
                omega_train = ~X_train.isnull() # matrix X_train with True in cells with evaluations and False in cells not rated
                glv = GroupLossVariance(X_train, omega_train, G, 1) #axis = 1 (0 rows e 1 columns)
                RgrpNR = glv.evaluate(X_est_train)
                rmse = RMSE(X_train, omega_train)
                result = rmse.evaluate(X_est_train)
                print("\nTrain :: X_train, omega_train :: X_est_train")
                print_results(dataset, algorithm, 0, 0, n_users, n_items, RgrpNR, result, original=True)

                glv = GroupLossVariance(X_train, omega_train, G, 1) #axis = 1 (0 rows e 1 columns)
                RgrpNR = glv.evaluate(X_gurobi)
                rmse = RMSE(X_train, omega_train)
                result = rmse.evaluate(X_gurobi)
                print("\nTrain :: X_train, omega_train :: X_gurobi")
                print_results(dataset, algorithm, h, i, n_users, n_items, RgrpNR, result, original=False)
                
                # --------------------------------------------------------------------------
                # Test
                omega_test = ~X_test.isnull() # matrix X_train with True in cells with evaluations and False in cells not rated
                glv = GroupLossVariance(X_test, omega_test, G, 1) #axis = 1 (0 rows e 1 columns)
                RgrpNR = glv.evaluate(X_est_train)
                rmse = RMSE(X_test, omega_test)
                result = rmse.evaluate(X_est_train)
                print("\nTest :: X_test, omega_test :: X_est_train")
                print_results(dataset, algorithm, 0, 0, n_users, n_items, RgrpNR, result, original=True)

                glv = GroupLossVariance(X_test, omega_test, G, 1) #axis = 1 (0 rows e 1 columns)
                RgrpNR = glv.evaluate(X_gurobi)
                rmse = RMSE(X_test, omega_test)
                result = rmse.evaluate(X_gurobi)
                print("\nTest :: X_test, omega_test :: X_gurobi")
                print_results(dataset, algorithm, h, i, n_users, n_items, RgrpNR, result, original=False)




df_experiment_results = pd.DataFrame(experiment_results)

# Verificar se o arquivo já existe
if os.path.exists('_results.csv'):
    # Ler o arquivo existente
    df_existing = pd.read_csv('_results.csv', sep=';')
    # Concatenar os dados existentes com os novos dados
    df_combined = pd.concat([df_existing, df_experiment_results], ignore_index=True)
else:
    # Se o arquivo não existir, os dados combinados são apenas os novos dados
    df_combined = df_experiment_results

# Salvar os dados combinados de volta no arquivo CSV
df_combined.to_csv('_results.csv', sep=';', index=False)

print('The results of the experiments were successfully saved in "_results.csv".".')
