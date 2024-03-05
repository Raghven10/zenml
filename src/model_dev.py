import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstarct class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):

        """
        Trains the model 

        Args: 
            X_train : Training data
            y_train: Training label

        Retunrs:
            None
    
        """

        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """

    def train(self, X_train, y_train, **kwargs):


        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training completed")
            return reg
        
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e



