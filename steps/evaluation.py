import pandas as pd
import logging
from sklearn.base import RegressorMixin
from typing_extensions import Tuple
from typing import Annotated
from zenml import step
from src.evaluation import MSE, R2Score, RMSE

@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "rmse"]
]:
    """Evaulating Model"""

    try:

        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2Score()
        r2 = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return r2, rmse
    
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise(e)
