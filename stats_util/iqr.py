import pandas as pd

class InterquartileRange:

    def __init__(self, df_col: pd.Series) -> None:
            self.data  = df_col

    def get_iqr_bounds(self, factor : float=1.5) -> tuple[float, float]:
        '''
        A function to calculate outlier bounds using the IQR methond
        Input: The IQR factor
        Output: The lower and upper outlier bounds
        '''
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3-Q1
        self.lower_iqr = Q1 - factor * IQR
        self.upper_iqr = Q3 + factor * IQR

        return self.lower_iqr, self.upper_iqr 
