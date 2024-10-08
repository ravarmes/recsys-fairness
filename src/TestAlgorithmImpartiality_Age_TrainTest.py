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

        age_00_17 = users_info[users_info['Age'] ==  1].index.intersection(list_users).tolist()
        age_18_24 = users_info[users_info['Age'] == 18].index.intersection(list_users).tolist()
        age_25_34 = users_info[users_info['Age'] == 25].index.intersection(list_users).tolist()
        age_35_44 = users_info[users_info['Age'] == 35].index.intersection(list_users).tolist()
        age_45_49 = users_info[users_info['Age'] == 45].index.intersection(list_users).tolist()
        age_50_55 = users_info[users_info['Age'] == 50].index.intersection(list_users).tolist()
        age_56_00 = users_info[users_info['Age'] == 56].index.intersection(list_users).tolist()

        G = {1: age_00_17, 2: age_18_24, 3: age_25_34, 4: age_35_44, 5: age_45_49, 6: age_50_55, 7: age_56_00}
        G_index = {1: [14, 194, 273, 132, 262], 2: [251, 175, 94, 86, 270, 48, 92, 96, 129, 107, 134, 64, 124, 288, 159, 26, 101, 61, 76, 126, 207, 191, 174, 168, 70, 215, 209, 171, 82, 149, 71, 33, 216, 157, 189, 23, 50, 231, 222, 201, 140, 163, 246, 244, 282, 265, 8, 90, 290, 158, 255, 237, 275], 3: [183, 200, 220, 212, 141, 230, 7, 60, 120, 170, 205, 16, 41, 145, 37, 34, 261, 69, 176, 32, 73, 35, 213, 133, 279, 285, 74, 276, 135, 252, 80, 225, 81, 63, 56, 138, 102, 295, 3, 296, 155, 161, 108, 51, 97, 269, 195, 283, 236, 21, 289, 232, 198, 66, 249, 238, 72, 293, 203, 188, 258, 10, 196, 139, 116, 291, 226, 233, 31, 142, 186, 118, 217, 151, 181, 45, 254, 42, 190, 89, 55, 147, 294, 169, 103, 93, 85, 28, 286, 39, 241, 267, 211, 223, 143, 229, 104, 110, 277, 136, 206, 40, 65, 11, 264, 29, 15, 22, 24, 234, 287, 130, 280, 131, 187, 164, 128, 43, 44, 298, 219, 75, 179, 109, 122, 173, 260, 202, 53, 79, 59, 253, 192, 204, 9, 119, 6, 111, 268, 240, 193, 106], 4: [272, 99, 299, 87, 199, 13, 84, 271, 1, 227, 18, 248, 117, 17, 127, 77, 177, 292, 156, 27, 214, 4, 5, 91, 297, 105, 49, 121, 98, 210, 52, 166, 95, 112, 88, 68, 208, 243, 152, 245, 228, 100, 250, 2, 278, 36, 144, 146, 256, 38, 57, 78, 172, 182, 150, 153, 25, 263], 5: [137, 221, 30, 160, 19, 67, 165, 113, 54, 281, 184, 167, 197, 125, 58, 47, 242, 83, 20, 235, 239, 148, 46, 62], 6: [247, 180, 266, 123, 257, 185, 114, 224, 178, 0, 115, 274], 7: [162, 12, 259, 154, 218, 284]}


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
                X_gurobi = AlgorithmImpartiality.make_matrix_X_gurobi_seven_groups(list_X_est, G, list_Zs) # calculate the recommendation matrix optimized by gurobi

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
