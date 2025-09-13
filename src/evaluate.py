import os
import json
import pandas as pd
import joblib
import logging
import yaml
from sklearn.metrics import r2_score,mean_squared_error
from dvclive import Live

root = os.path.dirname(os.path.dirname(__file__))
param_path=os.path.join(root,"params.yaml")

logger=logging.getLogger("evaluate")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler_path=os.path.join(root,"logs","evaluate.log")
file_handler=logging.FileHandler(file_handler_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    try:
        with open(params_path,"r") as f:
            params=yaml.safe_load(f)
            logger.debug(f"params safely loaded")
            return params
    except Exception as e:
        logger.error("failed to load params: {e}")
        raise

def load_model(params:dict):
    try:
        model_path=os.path.join(root,params["model"]["linear"])
        model=joblib.load(model_path)
        logger.debug(f"model loaded succesfully")
        return model
    except Exception as e:
        logger.error(f"failed to load model : {e}")
        raise    
def load_data(params:dict)->pd.DataFrame:
    try:
        data_path=os.path.join(root,params["dataset"]["test"])
        df=pd.read_csv(data_path)
        logger.debug(f"safely loaded dataset")
        return df
    except Exception as e:
        logger.error(f"failed to load data : {e}")
        raise

def evaluate(model,df:pd.DataFrame)->dict:
    try:
        x=df.drop(columns=["price"])
        y=df["price"]
        y_pred=model.predict(x)
        mse=mean_squared_error(y,y_pred)
        r2=r2_score(y,y_pred)
        metrics={"mse":mse,"r2":r2}
        logger.debug(f"evaluation is succesfully done")
        return metrics
    except Exception as e:
        logger.error(f"failed to evaluate : {e}")
        raise

def save_metrics(metrics: dict, params:dict):
    try:
        path=os.path.join(root,params["reports"]["metrics"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.debug(f"Metrics saved at {path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise    


def main():
    params=load_params(param_path)
    model=load_model(params)
    df=load_data(params)
    metrics=evaluate(model,df)
    save_metrics(metrics, params)

if __name__=="__main__":
    main()    