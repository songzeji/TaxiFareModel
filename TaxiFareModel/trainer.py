# imports
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
from TaxiFareModel import data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        
        time_pipe = Pipeline([
            ('TimeFeaturesEncoder', TimeFeaturesEncoder('pickup_datetime')),
            ('OneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])
                             
        dist_pipe = Pipeline([
            ('DistanceTransformer', DistanceTransformer()),
            ('StandardScaler',StandardScaler())
        ])
        
        time_features=['pickup_datetime']
        dist_features=["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']
        
        preprocessing = ColumnTransformer([
            ('time_pipe',time_pipe,time_features),
            ('dist_pipe',dist_pipe,dist_features),
        ],remainder='drop')
        
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('linear_regression',LinearRegression())
        ])
        self.pipeline=pipeline
        
    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred=self.pipeline.predict(X_test)
        rmse=np.sqrt(((y_pred-y_test)**2).mean())
        return rmse

if __name__ == "__main__":
    # get data
    df = data.get_data()
    # clean data
    df = data.clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y= df.fare_amount
    # hold out
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=.3)
    # train
    trainer=Trainer(X_train,y_train)
    trainer.set_pipeline()
    trainer.run()
    # evaluate
    print(trainer.evaluate(X_val,y_val))
