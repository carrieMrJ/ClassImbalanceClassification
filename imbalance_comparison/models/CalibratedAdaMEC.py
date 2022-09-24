# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:24:52 2017

@author: Nikolaos Nikolaou & Gavin Brown
"""
###############################################################################
# This code is licensed under the BSD 3-Clause License.
# It provides a template implementation of the Calibrated-AdaMEC
# method proposed in the paper:
#
# Cost-sensitive boosting algorithms: Do we really need them?,
# Nikolaos Nikolaou, Narayanan U. Edakunni, Meelis Kull,
# Peter A. Flach, Gavin Brown, Machine Learning, 104(2),
# pages 359-384, 2016 
# 
# This follows the pseudocode laid out on p15 of the supplementary
# material.  If you make use of this code, please cite the above paper.
#
# Thanks! Happy Boosting!
# Nikolaou et al.
#
###############################################################################

## Import packages
import numpy as np
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split # Instead of above package, use this with sklearn v. 0.18 and newer
from sklearn import tree  # If another sklearn classifier is used, remember to import it. Ignore warning.
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class CalibratedAdaMECClassifier:

    def __init__(self, base_learner, ensemble_size, C_FP, C_FN):
        """A Calibrated AdaMEC ensemble for cost-sensitive classification. 
        
            Parameters:
                
            base_learner: object, sklearn supported classifier, base learner to be used
                          e.g. decision stump: 'tree.DecisionTreeClassifier(max_depth=1)'
                          remember to import the relevant packages if other base learner
                          is used
                
            ensemble_size: integer,  AdaBoost ensemble (maximum) size 
            
            C_FP: float, cost of a single false positive (misclassifying a negative)
                
            C_FN: float, cost of a single false negative (misclassifying a positive)     
        """
        self.C_FP = C_FP
        self.C_FN = C_FN
        self.Pos = 0
        self.Neg = 0
        self.skew = 0
        self.UncalibratedAdaBoost = AdaBoostClassifier(base_estimator=base_learner, algorithm="SAMME",
                                                       n_estimators=ensemble_size)
        self.CalibratedAdaBoost = self.UncalibratedAdaBoost

    def fit(self, X, y):
        """Train calibrated AdaBoost ensemble -- also stores the class imbalance in the
		   training data to be later used for setting the minimum risk threshold for
		   predictions
    
              Parameters:
    		  
              x: array-like, shape (n_samples, n_features), training data
			  
              y: array-like, shape (n_samples), labels of training data        
         """
        # First compute class imbalance on training set.
        self.Pos = sum(y[np.where(y == 1)])  # Number of positive training examples
        self.Neg = len(y) - self.Pos  # Number of negative training examples

        # Now calculate the skew (combined asymmetry due to both cost and class imbalance)
        self.skew = self.C_FP * self.Neg / (self.C_FN * self.Pos + self.C_FP * self.Neg)

        # Reserve part of the training data for calibration
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.5)  # 50/50 train/calibration split

        # Train an (uncalibrated) AdaBoost ensemble
        self.UncalibratedAdaBoost.fit(X_train, y_train)

        # Now calibrate the ensemble on the data reserved for calibration
        # cv="prefit" means that the model is already fitted and only needs calibration
        self.CalibratedAdaBoost = CalibratedClassifierCV(self.UncalibratedAdaBoost, cv="prefit",
                                                         method="sigmoid")  # Uses Platt scaling (logistic sigmoid)
        # to calibrate
        self.CalibratedAdaBoost = self.CalibratedAdaBoost.fit(X_cal, y_cal)

        return self

    def predict(self, X_test):
        """Output AdaMEC (AdaBoost with shifted decision threshold) predictions
    
              Parameters:
    		  
              X_te: array-like, shape (n_samples, n_features), test data
    
              Returns:
    
              y_pred: array-like, shape (n_samples), predicted classes on test data     
         """
        scores = self.CalibratedAdaBoost.predict_proba(X_test)[:, 1]  # Positive Class scores

        y_pred = np.zeros(X_test.shape[0])
        y_pred[np.where(
            scores > self.skew)] = 1  # Classifications, AdaMEC uses a shifted decision threshold (skew-sensitive)

        return y_pred

    def predict_proba(self, X_test):
        """Output AdaMEC (AdaBoost with shifted decision threshold) predictions
    
              Parameters:
    
              X_te: array-like, shape (n_samples, n_features), test data
    
              Returns:
    
              scores: array-like, shape (n_samples), predicted scores (i.e.
                      calibrated probability estimates) for the positive
                      class for the test data     
         """
        proba = self.CalibratedAdaBoost.predict_proba(X_test)  # Class scores

        return proba
