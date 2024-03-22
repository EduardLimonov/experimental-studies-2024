import pandas as pd
import pickle
from typing import List
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame
from tsfresh.feature_extraction import settings


class SKPredModel:
    def __init__(self, model_path: str = "model.PICKLE"):
        """
        Here you initialize your model
        """
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

    def prepare_df(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Here you put any feature generation algorithms that you use in your model

        :param test_df:
        :return: test_df extended with generated features
        """

        test_df = test_df.reset_index()
        d = test_df.copy()
        d["Elapsed"] = d["Elapsed"].apply(self.__parse_elapsed)
        d["Timelimit"] = d["Timelimit"].apply(self.__parse_elapsed)
        d = self.__add_features(d)
        
        d.sort_values(["UID", "Submit"], inplace=True)

        d["Area"].fillna("?", inplace=True)
        for c in d.columns:
            if "mean_elapsed" in c:
                d[c].fillna(-1, inplace=True)
        
        d["Submit"] = pd.to_datetime(d["Submit"])
        d = self.__modify_df(d)

        predict_cols_needed = [
            "Area", "Partition", "ReqNodes", "ReqCPUS", "Timelimit", "Submit", "Priority", "UID"
        ] + \
        [c for c in d.columns if "mean_elapsed" in c or "Elapsed_" in c]
        d = d[predict_cols_needed]
        return d.loc[test_df.index]

    @property
    def model_keys(self) -> List[str]:
        return list(self._model.feature_names_)

    @staticmethod
    def __modify_df(df):
        df_exp = df.copy()
        df_exp = df_exp.sort_values("Submit")
        d_rolled = roll_time_series(df_exp[["UID", "Submit", "Elapsed"]], column_id="UID", column_sort="Submit", min_timeshift=1, max_timeshift=8)
        df_features = extract_features(
            d_rolled[["id", "Submit", "Elapsed"]], column_id="id", column_sort="Submit", 
            # default_fc_parameters=settings.MinimalFCParameters()
        ).reset_index().rename(columns={"level_0": "UID", "level_1": "Submit"})
    
        df_exp = df_exp.merge(df_features, on=["UID", "Submit"], how="left").fillna(-1)
        return df_exp
    
    @staticmethod
    def __add_features(d: pd.DataFrame) -> pd.DataFrame:
        gr = dict()
        for i in ['all', 4, 8, 16, 64, 256]:
            if i == 'all':
                gr[i]= d.groupby("UID").apply(lambda df: df["Elapsed"].expanding().mean().shift(1))
            else:
                gr[i]= d.groupby("UID").apply(lambda df: df["Elapsed"].rolling(window=i).mean().shift(1))
                
            
        d.set_index("UID", inplace=True)
        
        for uid in d.index.unique():
            for i in gr:
                d.loc[uid, f"mean_elapsed_{i}"] = gr[i].loc[uid].values
        
        d.reset_index(inplace=True)

        return d

    @staticmethod
    def __parse_elapsed(s: str) -> int:
        if "-" in s:
            days, rest = s.split('-')
            days = int(days)
        else:
            rest = s
            days = 0
    
        h, m, s = rest.split(":")
        h, m, s = int(h), int(m), int(s)
    
        return s + m * 60 + h * 60 * 60 + days * 24 * 60 * 60

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """
        Here you implement inference for your model

        :param test_df: dataframe to predict
        :return: vector of estimated times in milliseconds
        """
        for key in self.model_keys:
            assert key in test_df.keys(), f"{key} column missed in test_df"
            
        return self._model.predict(test_df)
