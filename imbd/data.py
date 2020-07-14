from typing import List
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, data_fp: str = 'data/0714train.csv'):
        self.data_fp = data_fp
        self.raw_df = pd.read_csv(data_fp)

    def shift_parser(self, shift) -> List:
        res = [0, 0]
        
        if not isinstance(shift ,str) and np.isnan(shift):
            return res

        shift_split = shift.split(';')
        
        if shift_split[0] == 'N':
            res[0] = 0
        elif shift_split[0] == 'D':
            res[0] = - float(shift_split[1])
        else:
            res[0] = float(shift_split[1])
            
        if shift_split[2] == 'N':
            res[1] = 0
        elif shift_split[0] == 'L':
            res[1] = - float(shift_split[3])
        else:
            res[1] = float(shift_split[3])
            
        return res

    def extract_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        shift_rng_1 = list(range(15, 39))
        shift_rng_2 = list(range(63, 83))
        shift_rng = shift_rng_1 + shift_rng_2
        coordinates = []
        
        for col in shift_rng:
            extracted = df[f'Input_C_{col:03d}'].apply(self.shift_parser).apply(pd.Series)
            extracted.columns = [f'Input_C_{col:03d}_Vertical', f'Input_C_{col:03d}_Horizon']
            coordinates.append(extracted)
            
        df = pd.concat(coordinates, axis=1)
        
        return df

    def build_stack_drill_df(self) -> pd.DataFrame:
        dfs, stack_drill_df = [], []
        
        for i in range(1, 7):
            df = self.raw_df.filter(regex=f"(Input_A{i}_*|Input_C_*)").join(self.raw_df[f"Output_A{i}"])

            for j in range(1, 25):
                df = df.rename(columns={f"Input_A{i}_{j:03d}": f"Input_A_{j:03d}"})
            
            dfs.append(df)

        for i, df in enumerate(dfs, start=1):
            df_ext = self.extract_shift(df)
            df_ext = pd.concat([df, df_ext], axis=1)
            df_ext = df_ext.drop([f'Input_C_{i:03d}' for i in range(15, 39)], axis=1)
            df_ext = df_ext.drop([f'Input_C_{i:03d}' for i in range(63, 83)], axis=1)
            df_ext = df_ext.rename(columns={f'Output_A{i}': "Output"})
            stack_drill_df.append(df_ext)

        stack_drill_df = pd.concat(stack_drill_df, axis=0)

        return stack_drill_df


    def build(self, data_type: str = 'stack') -> pd.DataFrame:
        if data_type == 'stack':
            return self.build_stack_drill_df()