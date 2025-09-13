import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,mean_squared_error,r2_score
import pandas as pd
import yaml
import joblib
import logging


root=os.path.dirname(os.path.dirname(__file__))
log_dir=os.path.join(root,"logs")
os.makedirs(log_dir,exist_ok=True)

# logger
logger=logging.getLogger("train")
logger.setLevel(logging.DEBUG)

# console handler
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# file handler
file_handler_path=os.path.join(log_dir,"train_logs.log")
file_handler=logging.FileHandler(file_handler_path)
file_handler.setLevel(logging.DEBUG)

# formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(param_path:str)->dict:
    try:
        with open(param_path,"r")as f:
            params=yaml.safe_load(f)
            logger.debug(f"params for training loaded succesfully ")
            return params
    except Exception as e:
        logger.error(f"params failed to load for training model the message is {e}")
        raise    
def load_data(data_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        logger.debug(f"loaded data for training succesfully")
        return df
    except Exception as e:
        logger.error(f"data load for training failed from {data_path}")
        raise

def preprocess(df:pd.DataFrame,params:dict):
    test_size=params.get("train",{}).get("test_size",0.2)
    random_state=params.get("train",{}).get("random_state",42)
    x=df.drop(columns="price")
    y=df["price"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    return model,mse,r2

def save_model(model:LinearRegression,model_dir:str):
    try:
        os.makedirs(model_dir,exist_ok=True)
        model_path=os.path.join(model_dir,"linear.pkl")
        joblib.dump(model,model_path)
        logger.debug(f"model saved in {model_path}")
        return
    except Exception as e:
        logger.error("model could not be saved the error is {e}")


def main():
    params_path=os.path.join(root,"params.yaml")
    model_path=os.path.join(root,"models")
    params=load_param(params_path)
    processed_path=os.path.join(root,params["dataset"]["processed_data"])
    df=load_data(processed_path)
    model,mse,r2=preprocess(df,params)
    print("mse is ", mse)
    print("r2 score is ",r2)
    save_model(model,model_path)

if __name__=="__main__":
    main()   