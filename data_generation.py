# General imports.
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
import math
from sklearn.preprocessing import StandardScaler


class DataGeneration():
    
    
    def generate_linear_gaussian(self,
                                 n: int = 1000, 
                                 x_causes_y: bool = True,
                                 xy_coeff: float = 1.0,
                                 m_structure: bool = False,
                                 butterfly_structure: bool = False,
                                 coefficient_range: tuple = (1.25, 2.5),
                                 verbose: bool = False):

        '''
        '''

        # Construct noise terms of structural equation.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        total_vars = 10
        if m_structure:
            total_vars += 3
        if butterfly_structure:
            total_vars += 3
        noise = []
        for var in range(total_vars):
            noise.append(np.random.normal(loc = 0.0, scale = 1.0, size = n).reshape(-1, 1))

        # Define coefficient generator.
        coeff = lambda : 1
        
        # Define variables.
        if not x_causes_y:
            xy_coeff = 0
        Z1 = noise[2]
        Z4 = noise[5]
        Z5 = noise[6]
        Z8 = noise[9]
        X = coeff()*Z1 + coeff()*Z5 + noise[0]
        if m_structure:
            M1 = noise[10]
            M2 = noise[11]
            M3 = coeff()*M1 + coeff()*M2 + noise[12]
            X = X + coeff()*M1
        if butterfly_structure:
            if m_structure:
                B1 = noise[13]
                B2 = noise[14]
                B3 = coeff()*B1 + coeff()*B2 + noise[15]
            else:
                B1 = noise[10]
                B2 = noise[11]
                B3 = coeff()*B1 + coeff()*B2 + noise[12]
            X = X + coeff()*B1 + coeff()*B3
        Z3 = coeff()*X + noise[4]
        Y = xy_coeff*X + coeff()*Z1 + coeff()*Z3 + coeff()*Z4 + noise[1]
        if m_structure:
            Y = Y + coeff()*M2
        if butterfly_structure:
            Y = Y + coeff()*B2 + coeff()*B3
        Z2 = coeff()*X + coeff()*Y + noise[3]
        Z6 = coeff()*Y + noise[7]
        Z7 = coeff()*X + noise[8]
            
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z2": Z2.reshape(-1), 
                                "Z3": Z3.reshape(-1), 
                                "Z4": Z4.reshape(-1), 
                                "Z5": Z5.reshape(-1),
                                "Z6": Z6.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1)})
        df_noise = pd.DataFrame({"X": noise[0].reshape(-1), 
                                 "Y": noise[1].reshape(-1), 
                                 "Z1": noise[2].reshape(-1),
                                 "Z2": noise[3].reshape(-1), 
                                 "Z3": noise[4].reshape(-1), 
                                 "Z4": noise[5].reshape(-1), 
                                 "Z5": noise[6].reshape(-1),
                                 "Z6": noise[7].reshape(-1), 
                                 "Z7": noise[8].reshape(-1), 
                                 "Z8": noise[9].reshape(-1)})
        var_names = ["Z" + str(i) for i in range(1, 9)]
        var_names = ["X", "Y"] + var_names
        
        if m_structure:
            df_vars["M1"]  = M1.reshape(-1)
            df_vars["M2"]  = M2.reshape(-1)
            df_vars["M3"]  = M3.reshape(-1)
            df_noise["M1"] = M1.reshape(-1)
            df_noise["M2"] = M2.reshape(-1)
            df_noise["M3"] = M3.reshape(-1)

        if butterfly_structure:
            df_vars["B1"]  = B1.reshape(-1)
            df_vars["B2"]  = B2.reshape(-1)
            df_vars["B3"]  = B3.reshape(-1)
            df_noise["B1"] = B1.reshape(-1)
            df_noise["B2"] = B2.reshape(-1)
            df_noise["B3"] = B3.reshape(-1)
        
        if verbose:
            print("VARIABLES:")
            display(df_vars.head())
            print("NOISE:")
            display(df_noise.head())

        return df_vars, xy_coeff
    