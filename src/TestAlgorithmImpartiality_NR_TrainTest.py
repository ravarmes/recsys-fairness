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
datasets = ['MovieLens-1M', 'Goodbooks-10k', 'Songs']

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
hs = [3, 5, 10, 15, 20, 23]
# hs = [15, 20, 23]
# hs = [23]
# hs = [3]


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
        # G group: identifying the groups (NR: users grouped by number of ratings for available items)
        # G = {1: advantaged_group, 2: disadvantaged_group}
        # Obter a lista de usuários ordenada por número de avaliações
        sorted_users = X_est_train.index.tolist()

        # Definir os grupos com base no número de usuários especificado
        num_advantaged = math.ceil(0.05 * n_users)

        advantaged_group = sorted_users[:num_advantaged]
        disadvantaged_group = sorted_users[num_advantaged:]

        # Criar o dicionário G com os grupos
        G = {1: advantaged_group, 2: disadvantaged_group}
        # print(G)

        # Criar o dicionário G_index com os índices dos grupos
        G_index = {1: list(range(num_advantaged)), 2: list(range(num_advantaged, len(sorted_users)))}
        # print(G_index)


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
