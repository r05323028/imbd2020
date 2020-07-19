from typing import List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imbd.preprocessing import FeaturesSelector, QuantizationTransformer, FillNATransformer, OutlierDetector, VarianceFeatureSelector


class DataLoader:
    labels = [
        'Input_A6_024', 'Input_A3_016', 'Input_C_013', 'Input_A2_016',
        'Input_A3_017', 'Input_C_050', 'Input_A6_001', 'Input_C_096',
        'Input_A3_018', 'Input_A6_019', 'Input_A1_020', 'Input_A6_011',
        'Input_A3_015', 'Input_C_046', 'Input_C_049', 'Input_A2_024',
        'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
    ]
    shift_cols = [f'Input_C_{i:03d}' for i in range(15, 39)
                  ] + [f'Input_C_{i:03d}' for i in range(63, 83)]
    drill_cols = [f'Output_A{i}' for i in range(1, 7)]

    def __init__(self, data_fp: str = 'data/0714train.csv'):
        self.data_fp = data_fp
        self.raw_df = pd.read_csv(data_fp)

    def shift_parser(self, shift) -> List:
        res = [0, 0]

        if not isinstance(shift, str) and np.isnan(shift):
            return res

        shift_split = shift.split(';')

        if shift_split[0] == 'N':
            res[1] = 0
        elif shift_split[0] == 'D':
            res[1] = -float(shift_split[1])
        else:
            res[1] = float(shift_split[1])

        if shift_split[2] == 'N':
            res[0] = 0
        elif shift_split[0] == 'L':
            res[0] = -float(shift_split[3])
        else:
            res[0] = float(shift_split[3])

        return res

    def extract_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        shift_rng_1 = list(range(15, 39))
        shift_rng_2 = list(range(63, 83))
        shift_rng = shift_rng_1 + shift_rng_2
        coordinates = []

        for col in shift_rng:
            extracted = df[f'Input_C_{col:03d}'].apply(
                self.shift_parser).apply(pd.Series)
            extracted.columns = [
                f'Input_C_{col:03d}_x', f'Input_C_{col:03d}_y'
            ]
            coordinates.append(extracted)

        df = pd.concat(coordinates, axis=1)

        return df

    def build_stack_drill_df(self) -> pd.DataFrame:
        dfs, stack_drill_df = [], []

        for i in range(1, 7):
            df = self.raw_df.filter(regex=f"(Input_A{i}_*|Input_C_*)").join(
                self.raw_df[f"Output_A{i}"])

            for j in range(1, 25):
                df = df.rename(
                    columns={f"Input_A{i}_{j:03d}": f"Input_A_{j:03d}"})

            df['drill_number'] = i

            dfs.append(df)

        for i, df in enumerate(dfs, start=1):
            df_ext = self.extract_shift(df)
            df_ext = pd.concat([df, df_ext], axis=1)
            df_ext = df_ext.drop([f'Input_C_{i:03d}' for i in range(15, 39)],
                                 axis=1)
            df_ext = df_ext.drop([f'Input_C_{i:03d}' for i in range(63, 83)],
                                 axis=1)
            df_ext = df_ext.rename(columns={f'Output_A{i}': "Output"})
            stack_drill_df.append(df_ext)

        stack_drill_df = pd.concat(stack_drill_df, axis=0)

        # dummies = pd.get_dummies(stack_drill_df['drill_number'], prefix='Drill')
        # stack_drill_df = pd.concat([stack_drill_df, dummies], axis=1)
        # stack_drill_df = stack_drill_df.drop('drill_number', axis=1)

        return stack_drill_df

    def build_label_20_df(self) -> pd.DataFrame:
        df = self.raw_df.filter(regex=f"(Input_A[0-9]+_*|Input_C_*|Output_*)")
        df_ext = self.extract_shift(df)
        df_ret = pd.concat([df, df_ext], axis=1)
        df_ret = df_ret.drop(self.shift_cols, axis=1)
        label_not_na_rows = df_ret[self.labels].dropna().index

        return df_ret.iloc[label_not_na_rows].reset_index(drop=True)

    def build(self, data_type: str = 'stack') -> pd.DataFrame:
        if data_type == 'stack':
            return self.build_stack_drill_df()


class DataPreprocessor:
    def __call__(self, df) -> pd.DataFrame:
        pipe = Pipeline(steps=[
            ('features_select', FeaturesSelector()),
            ('quantization', QuantizationTransformer()),
            ('fill_na', FillNATransformer()),
            ('variance_selector', VarianceFeatureSelector()),
            ('outlier_detection', OutlierDetector()),
        ],
                        verbose=True)
        out_df = pipe.fit_transform(df)

        return out_df