import random as random
from sklearn.metrics import mean_squared_error
import gurobipy as gp
import pandas as pd
import numpy as np
from AlgorithmUserFairness import IndividualLossVariance

class AlgorithmImpartiality():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
    
    def evaluate(self, X_est, h):
        list_X_est = []
        for x in range(0, h):
            print("x: ", x)
            list_X_est.append(self.get_X_est(X_est.copy()))
        return list_X_est
        
    def get_differences_means(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)

        E = (X_est - X)
        losses = E.mean(axis=self.axis)
        return losses

    def get_differences_vars(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        losses = E.mean(axis=self.axis)
        return losses
        
    # Version 02 of the strategy based on mean and loss (WITH concern for the sign)
    def get_X_est(self, X_est):
        list_dif_mean = self.get_differences_means(X_est)
        list_dif_var = self.get_differences_vars(X_est)

        list_dif_var = list_dif_var / 4

        # Converting to NumPy arrays for vectorized operation
        list_dif_mean = list_dif_mean.to_numpy()
        list_dif_var = list_dif_var.to_numpy()
        
        # Determine lower and upper bounds for generating random values
        lower_bounds = np.where(list_dif_mean > 0, 0, list_dif_var)
        upper_bounds = np.where(list_dif_mean > 0, list_dif_var, 0)
        
        # Adjust the case where the lower bound is greater than the upper bound
        lower_bounds = np.expand_dims(lower_bounds, axis=1)
        upper_bounds = np.expand_dims(upper_bounds, axis=1)

        # Generate a single random value per row within the defined bounds
        random_values = np.random.uniform(lower_bounds, upper_bounds, size=(X_est.shape[0], 1))

        # Repeat the random value per row for all columns
        random_values = np.tile(random_values, X_est.shape[1])

        # Multiply by the opposite sign of list_dif_mean to respect the original opposite sign
        sign_adjustment = -np.sign(list_dif_mean)[:, np.newaxis]
        random_values *= sign_adjustment

        # Update X_est with random values, ensuring values are between 1 and 5
        X_est = X_est.add(random_values)
        X_est = X_est.clip(lower=1, upper=5)
        
        return X_est
    
    
    def losses_to_Z(list_losses, n_users=300):
        Z = []
        row = []
        for i in range(0, n_users):
            for losses in list_losses:
                row.append(losses.values[i])
            Z.append(row.copy())
            row.clear()
        return Z

    def matrices_Zs(Z, G): # return a Z matrix for each group
        list_Zs = []
        for group in G: # G = {1: [1,2], 2: [3,4,5]}
            Z_ = []
            list_users = G[group]
            for user in list_users:
                Z_.append(Z[user].copy())   
            list_Zs.append(Z_)
        return list_Zs
    

    def make_matrix_X_gurobi(list_X_est, G, list_Zs):

        # User labels and Rindv (individual injustices)
        users_g1 = ["U{}".format(user) for user in G[1]]  
        ls = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g2 = ["U{}".format(user) for user in G[2]]

        # Initialize the model
        m = gp.Model()

        # Decision variables
        x1 = m.addVars(users_g1, ls, vtype=gp.GRB.BINARY, name="x1")
        x2 = m.addVars(users_g2, ls, vtype=gp.GRB.BINARY, name="x2")

        # Dictionary with individual losses
        Z1 = list_Zs[0]
        Z2 = list_Zs[1]

        # Objective function
        L1 = gp.quicksum(Z1[i][j] * x1[users_g1[i], ls[j]] for i in range(len(users_g1)) for j in range(len(ls))) / len(users_g1)
        L2 = gp.quicksum(Z2[i][j] * x2[users_g2[i], ls[j]] for i in range(len(users_g2)) for j in range(len(ls))) / len(users_g2)
        LMean = (L1 + L2) / 2
        Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2) / 2

        m.setObjective(Rgrp, sense=gp.GRB.MINIMIZE)

        # Constraints to ensure all users have a Rindv (individual injustice calculation)
        m.addConstrs((x1.sum(i, '*') == 1 for i in users_g1), name="c1")
        m.addConstrs((x2.sum(i, '*') == 1 for i in users_g2), name="c2")

        # Run the model
        m.optimize()

        matrix_final = []
        indices = list_X_est[0].index.tolist()

        for i in indices:
            if i in G[1]:
                for j in ls:
                    if round(x1['U' + str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        break  # Stop searching once we find the match
            elif i in G[2]:
                for j in ls:
                    if round(x2['U' + str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        break  # Stop searching once we find the match

        X_gurobi = pd.DataFrame(matrix_final, columns=list_X_est[0].columns.values)
        X_gurobi['UserID'] = indices
        X_gurobi.set_index("UserID", inplace=True)

        return X_gurobi



    # initial version (unrefactored - just for seven groups)
    def make_matrix_X_gurobi_seven_groups(list_X_est, G, list_Zs):

        # User labels and Rindv (individual injustices)
        users_g1 = ["U{}".format(user) for user in G[1]]  
        ls_g1 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g2 = ["U{}".format(user) for user in G[2]] 
        ls_g2 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g3 = ["U{}".format(user) for user in G[3]] 
        ls_g3 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g4 = ["U{}".format(user) for user in G[4]] 
        ls_g4 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g5 = ["U{}".format(user) for user in G[5]] 
        ls_g5 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g6 = ["U{}".format(user) for user in G[6]] 
        ls_g6 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 
        users_g7 = ["U{}".format(user) for user in G[7]] 
        ls_g7 = ["l{}".format(j + 1) for j in range(len(list_X_est))] 

        # Dictionary with individual losses
        Z1 = list_Zs[0]
        preferences1 = dict()
        for i, user in enumerate(users_g1):
            for j, l in enumerate(ls_g1):
                preferences1[user, l] = Z1[i][j]

        Z2 = list_Zs[1]
        preferences2 = dict()
        for i, user in enumerate(users_g2):
            for j, l in enumerate(ls_g2):  
                preferences2[user, l] = Z2[i][j]

        Z3 = list_Zs[2]
        preferences3 = dict()
        for i, user in enumerate(users_g3):
            for j, l in enumerate(ls_g3):  
                preferences3[user, l] = Z3[i][j]

        Z4 = list_Zs[3]
        preferences4 = dict()
        for i, user in enumerate(users_g4):
            for j, l in enumerate(ls_g4):  
                preferences4[user, l] = Z4[i][j]

        Z5 = list_Zs[4]
        preferences5 = dict()
        for i, user in enumerate(users_g5):
            for j, l in enumerate(ls_g5):  
                preferences5[user, l] = Z5[i][j]

        Z6 = list_Zs[5]
        preferences6 = dict()
        for i, user in enumerate(users_g6):
            for j, l in enumerate(ls_g6):  
                preferences6[user, l] = Z6[i][j]

        Z7 = list_Zs[6]
        preferences7 = dict()
        for i, user in enumerate(users_g7):
            for j, l in enumerate(ls_g7):  
                preferences7[user, l] = Z7[i][j]


        # Initialize the model
        m = gp.Model()

        # Decision variables
        x1 = m.addVars(users_g1, ls_g1, vtype=gp.GRB.BINARY)
        x2 = m.addVars(users_g2, ls_g2, vtype=gp.GRB.BINARY)
        x3 = m.addVars(users_g3, ls_g3, vtype=gp.GRB.BINARY)
        x4 = m.addVars(users_g4, ls_g4, vtype=gp.GRB.BINARY)
        x5 = m.addVars(users_g5, ls_g5, vtype=gp.GRB.BINARY)
        x6 = m.addVars(users_g6, ls_g6, vtype=gp.GRB.BINARY)
        x7 = m.addVars(users_g7, ls_g7, vtype=gp.GRB.BINARY)


        # Objective function
        # In this case, the objective function seeks to minimize the variance between the injustices of the groups (Li) 
        # Li can also be understood as the average of the individual injustices of group i. 
        # Rgrp: the variance of all the injustices of the groups (Li).

        L1 = x1.prod(preferences1)/len(users_g1)
        L2 = x2.prod(preferences2)/len(users_g2)
        L3 = x3.prod(preferences3)/len(users_g3)
        L4 = x4.prod(preferences4)/len(users_g4)
        L5 = x5.prod(preferences5)/len(users_g5)
        L6 = x6.prod(preferences6)/len(users_g6)
        L7 = x7.prod(preferences7)/len(users_g7)
        LMean = (L1 + L2 + L3 + L4 + L5 + L6 + L7) / 7
        Rgrp = ((L1 - LMean)**2 + (L2 - LMean)**2 + (L3 - LMean)**2 + (L4 - LMean)**2 + (L5 - LMean)**2 + (L6 - LMean)**2 + (L7 - LMean)**2)/7

        m.setObjective(Rgrp, sense=gp.GRB.MINIMIZE)

        # Constraints to ensure all users will have a Rindv (individual injustice calculation)
        c1 = m.addConstrs(x1.sum(i, '*') == 1 for i in users_g1)
        c2 = m.addConstrs(x2.sum(i, '*') == 1 for i in users_g2)
        c3 = m.addConstrs(x3.sum(i, '*') == 1 for i in users_g3)
        c4 = m.addConstrs(x4.sum(i, '*') == 1 for i in users_g4)
        c5 = m.addConstrs(x5.sum(i, '*') == 1 for i in users_g5)
        c6 = m.addConstrs(x6.sum(i, '*') == 1 for i in users_g6)
        c7 = m.addConstrs(x7.sum(i, '*') == 1 for i in users_g7)

        # Run the model
        m.optimize()

        matrix_final = []
        indices = list_X_est[0].index.tolist()
        qtd_g1 = 0
        qtd_g2 = 0
        qtd_g3 = 0
        qtd_g4 = 0
        qtd_g5 = 0
        qtd_g6 = 0
        qtd_g7 = 0

        for i in indices:
            if i in G[1]:
                for j in ls_g1:
                    if round(x1['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g1 += 1
            if i in G[2]:
                for j in ls_g2:
                    if round(x2['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g2 += 1
            if i in G[3]:
                for j in ls_g3:
                    if round(x3['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g3 += 1
            if i in G[4]:
                for j in ls_g4:
                    if round(x4['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g4 += 1
            if i in G[5]:
                for j in ls_g5:
                    if round(x5['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g5 += 1
            if i in G[6]:
                for j in ls_g6:
                    if round(x6['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g6 += 1
            if i in G[7]:
                for j in ls_g7:
                    if round(x7['U'+str(i), j].X) == 1:
                        matrix = list_X_est[int(j[1:]) - 1]
                        matrix_final.append(matrix.loc[i].tolist())
                        qtd_g7 += 1

        X_gurobi = pd.DataFrame(matrix_final, columns=list_X_est[0].columns.values)
        X_gurobi['UserID'] = indices
        X_gurobi.set_index("UserID", inplace=True)
        
        return X_gurobi
