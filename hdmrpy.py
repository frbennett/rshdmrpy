"""
*******************************************************************************
Global sensitivity analysis using a Sparse Random Sampling - High Dimensional 
Model Representation (HDMR) using the Group Method of Data Handling (GMDH) for 
parameter selection and linear regression for parameter refinement
*******************************************************************************

author: 'Frederick Bennett'

"""
import pandas as pd
import math
import numpy as np
import scipy.special as sp
from gmdhpy.gmdh import Regressor
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV 
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
from sklearn import metrics
from scipy.stats import linregress 
from numba import jit

class rshdmr():
    
    def __init__(self,data_file, poly_order=4 ,**kwargs):
        self._seq_type='mode1'
        self._poly_order = poly_order 
        self._gmdh_ref_functions = 'linear_cov'
        self._admix_features = True
        self._alpha_ridge = 0.5
        self._alpha_lasso = 0.001
        self._epsilon = 0.001
        self._cutoff = 0.0001
        self._regression_type = 'lasso'
        self._criterion_type='validate'
        self._hdmr_order = 2
        self._index_cutoff = 0.01
        self._manual_best_neurons_selection=False 
        self._min_best_neurons_count=20
        self._n_jobs = 1
        for key, value in kwargs.items():
            setattr(self, "_"+key, value)
        self.read_data(data_file)

        
    def read_data(self,data_file):
        """
        dsdsd
        """
        if isinstance(data_file, pd.DataFrame):
            print(' found a dataframe')
            df = data_file
        if isinstance(data_file, str):
            df = pd.read_csv(data_file)
        self.Y = df['Y']
        self.X = df.drop('Y', axis=1)
        # we can clean up the original dataframe
        del df
        
    def shift_legendre(self,n,x):
        funct = math.sqrt(2*n+1) * sp.eval_sh_legendre(n,x)
        return funct
    
        
    def transform_data(self):
        self.X_T = pd.DataFrame()
        self.ranges = {}
        feature_names = list(self.X.columns.values)
        print(feature_names)
        for column in feature_names:
            max = self.X[column].max()
            min = self.X[column].min()
            print(column + " : min " + str(min) + " max " + str(max)) 
            self.X_T[column] = (self.X[column] - min) / (max-min)
            self.ranges[column] = [min,max]
        
            
    def legendre_expand(self):
        self.primitive_variables = []
        self.poly_orders = []
        self.X_T_L = pd.DataFrame()
        for column in self.X_T:
            for n in range (1,self._poly_order+1):
                self.primitive_variables.append(column)
                self.poly_orders.append(n)
                column_heading = column + "_" + str(n)
                self.X_T_L[column_heading] = [self.shift_legendre(n, x) for x in self.X_T[column]]
        self.exp_feature_names = list(self.X_T_L.columns.values) 
        
    def gmdh_regression(self):
        self.gmdh_model = Regressor(ref_functions=(self._gmdh_ref_functions),
                      criterion_type= self._criterion_type,
                      feature_names=self.exp_feature_names,
                      criterion_minimum_width=5,
                      stop_train_epsilon_condition=self._epsilon,
                      layer_err_criterion='top',
                      l2=0.5,                                    
                      seq_type= self._seq_type , 
                      max_layer_count= 50,
                      normalize=True,
                      keep_partial_neurons = False,
                      admix_features = self._admix_features,
                      manual_best_neurons_selection = self._manual_best_neurons_selection, 
                      min_best_neurons_count = self._min_best_neurons_count,
                      n_jobs=self._n_jobs)
        self.gmdh_model.fit(self.X_T_L, self.Y)

        selected_features = len(self.gmdh_model.get_selected_features_indices())
        print("selected features ", selected_features)
        print("=============================================")
        self.data = pd.DataFrame()
        selected_indices = self.gmdh_model.get_selected_features_indices()
        feature_count = len(self.exp_feature_names)
        self.selected_list = []
        self.primitive_list = []
        for order in range(1, self._hdmr_order + 1):
            for combo in combinations(selected_indices, order):
                header = ''
                series = []
                primitive_name = []
                derived_name = []
                for i in combo:
                    if header == '':
                        header = self.exp_feature_names[i]
                        series = self.X_T_L[self.exp_feature_names[i]]
                        primitive_name.append(self.primitive_variables[i])
                        derived_name.append(self.exp_feature_names[i])
                    else:
                        header = header + '*' + self.exp_feature_names[i]
                        feature_name = self.exp_feature_names[i]
                        series = series * self.X_T_L[self.exp_feature_names[i]]
                        primitive_name.append(self.primitive_variables[i])
                        derived_name.append(self.exp_feature_names[i])
                duplicates = pd.Series(primitive_name)[pd.Series(primitive_name).duplicated()].values
                result = 'NO duplicates'
                if len(duplicates) > 0:
                    result = 'duplicates'
                else:
                    self.data[header] = series
                    self.selected_list.append(derived_name)
                    self.primitive_list.append(primitive_name)
    
 
    def ridge_regression(self, **kwargs):
        if self._regression_type == 'lasso' :
            self.ridgereg = LassoCV(max_iter=50000)
            #self.ridgereg = LassoCV(max_iter=1e5, cv=10)
            self.ridgereg.fit(self.data, self.Y)
        elif self._regression_type == 'ard' :
            self.ridgereg = ARDRegression() 
            self.ridgereg.fit(self.data, self.Y)  
        elif self._regression_type == 'elastic' :
            self.ridgereg = ElasticNetCV(cv=10)
            self.ridgereg.fit(self.data, self.Y)
        elif self._regression_type == 'lars' :
            self.ridgereg = LarsCV(cv=10) 
            self.ridgereg.fit(self.data, self.Y)
        elif self._regression_type == 'lassolars' :
            self.ridgereg =  LassoLarsCV(cv=5)
            self.ridgereg.fit(self.data, self.Y)
        elif self._regression_type == 'ordinary' :
            self.ridgereg =  LinearRegression() 
            self.ridgereg.fit(self.data, self.Y)
        elif self._regression_type == 'ridge' :
            self.ridgereg =  RidgeCV() 
            self.ridgereg.fit(self.data, self.Y)
            
            
                    
    def eval_sobol_indices(self):
        total_variance = np.var(self.Y)
        self.sobol_indexes = pd.DataFrame(columns=['index', 'value'])
        total_coeff_squared = 0
        for i in range(0, len(self.primitive_list)):
            total_coeff_squared += self.ridgereg.coef_[i] * self.ridgereg.coef_[i]
        print('total coeff squared : ',total_coeff_squared)
        print('total variance : ',total_variance)   

        a = self.primitive_list
        b = []
        for i in a:
            if sorted(i) not in b:
                b.append(sorted(i))

        for unique in b:
            key = ''
            for variable_name in unique:
                key += ',' + variable_name
            key = key[1:]

            coeff_squared = 0
            for i in range(0, len(self.primitive_list)):
                if sorted(self.primitive_list[i]) == sorted(unique):
                    coeff_squared += self.ridgereg.coef_[i] * self.ridgereg.coef_[i]
            # index = coeff_squared / total_coeff_squared
            index = coeff_squared / total_variance 
            self.sobol_indexes.loc[len(self.sobol_indexes)] = [key, index]

    def predict(self,X):
        sum = self.ridgereg.intercept_
        primitives = list(self.X.columns.values)
        X_expanded = {}
        for i in range(0, len(X)):
            # Transform input
            min = self.ranges[primitives[i]][0]
            max = self.ranges[primitives[i]][1]
            X_T = (X[i] - min) / (max - min)
            for j in range(1, self._poly_order + 1):
                label = primitives[i] + '_' + str(j)
                legendre = self.shift_legendre(j, X_T)
                X_expanded[label] = legendre
        # print(X_expanded)

        sum = self.ridgereg.intercept_
        for i in range(0, len(self.ridgereg.coef_)):
            coeff = self.ridgereg.coef_[i]
            product = 1
            terms = self.selected_list[i]
            for term in terms:
                product *= X_expanded[term]
            sum += coeff * product
        return sum

    def evaluate_func(self,X):
        sum = self.ridgereg.intercept_
        primitives = list(self.X.columns.values)
        X_expanded ={}
        for i in range(0, len(X)):
            # Transform input
            min = self.ranges[primitives[i]][0]
            max = self.ranges[primitives[i]][1]
            X_T = (X[i] - min) / (max-min)
            for j in range(1, self._poly_order+1):
                label = primitives[i] +'_' + str(j)
                legendre = self.shift_legendre(j,X_T)
                X_expanded[label] = [legendre]
    
        for key in self.ridge_coeffs:
            gmdh_coeff = self.selected_features_dict[key]
            ridge_coeff = self.ridge_coeffs[key][1]
            if len(gmdh_coeff)==3:    
                variable_term = X_expanded[gmdh_coeff[0]][0]
                sum += variable_term * ridge_coeff
            else:
                variable_term = X_expanded[gmdh_coeff[0]][0] * X_expanded[gmdh_coeff[1]][0]
                sum += variable_term * ridge_coeff         
        return sum
    
    def plot_hdmr(self):
        y_pred = self.ridgereg.predict(self.data)
        matplotlib.pyplot.scatter(self.Y,y_pred)
        matplotlib.pyplot.ylabel('Predicted')
        matplotlib.pyplot.xlabel('Experimental')
        matplotlib.pyplot.show()
        
    def stats(self):
        y_pred = self.ridgereg.predict(self.data)
        mse = metrics.mean_squared_error(y_pred,self.Y)
        mae = metrics.mean_absolute_error(y_pred,self.Y)
        evs = metrics.explained_variance_score(y_pred,self.Y)
        slope, intercept, r_value, p_value, std_err = linregress(self.Y, y_pred)
        print("mae error on test set   : {mae:0.3f}".format(mae=mae))
        print("mse error on test set   : {mse:0.3f}".format(mse=mse))
        print("explained variance score: {evs:0.3f}".format(evs=evs))
        print("===============================")
        print("slope     : ", slope)
        print("r value   : ", r_value)
        print("r^2       : ", r_value*r_value)
        print("p value   : ", p_value)
        print("std error : ", std_err)
        
    def print_sobol_indices(self):
        self.eval_sobol_indices()
        for i, row in self.sobol_indexes.iterrows():
            if row['value'] > self._index_cutoff :
                print(row['index'], ' : ', row['value'])
                
    def auto(self):
        self.transform_data()
        self.legendre_expand()
        print('====================================')
        self.gmdh_regression()
        print('====================================')
        self.ridge_regression()
        self.print_sobol_indices()
        self.plot_hdmr()
        

