import pandas as pd
import os
import logging
import yaml

# setting log and root folder

root=os.path.dirname(os.path.dirname(__file__))
log_dir=os.path.join(root,"logs")
os.makedirs(log_dir,exist_ok=True)

# logger 
logger=logging.getLogger("preprocess")
logger.setLevel(logging.DEBUG)

# console handler 
console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# file handler 
log_filehandler_path=os.path.join(log_dir,"preprocess.log")
file_handler=logging.FileHandler(log_filehandler_path)
file_handler.setLevel(logging.DEBUG)

# formatter 
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# laod params
def load_params(params_path:str)->dict:
    try:
        with open (params_path,"r")as f:
            params=yaml.safe_load(f)
        logger.debug(f"safely loaded the params.yaml")
        return params 
    except Exception as e:
        logger.error(f"failed to load params.yaml from path{params_path}")
        raise

# load data
def load_data(data_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        logger.debug(f"safely loaded data from path{data_path}")
        return df
    except Exception as e:
        logger.error(f"data load failed from path {data_path}")
        raise

# pre_processing data 
def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df=df.fillna(df.median(numeric_only=True))
        logger.debug(f"safely preprocessed data")
        return df
    except Exception as e:
        logger.error(f"some error occured during preprocessing{e}")
        raise

# save preprocess_data file 
def save_preprocessed_data(df:pd.DataFrame,data_path:str)->None:
    try:
        os.makedirs(os.path.dirname(data_path),exist_ok=True)
        df.to_csv(data_path,index=False)
        logger.debug(f"processed data saved succesfully")
        return
    except Exception as e:
        logger.error(f"preprocessed data not saved at {data_path}")
        raise

def main():
    params_path=os.path.join(root,"params.yaml")
    params=load_params(params_path)
    data_path=os.path.join(root,params["dataset"]["path"])
    processed_path=os.path.join(root,params["dataset"]["processed_data"])

    df=load_data(data_path)
    processed_df=preprocess_data(df)
    save_preprocessed_data(processed_df,processed_path)

if __name__ == "__main__":
    main()