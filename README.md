# IMBD 2020

## Performance

### 2020-07-19

#### Steps

```python
Pipeline(steps=[('features_select',
                 <imbd.preprocessing.FeaturesSelector object at 0x10d5dad30>),
                ('quantization',
                 <imbd.preprocessing.QuantizationTransformer object at 0x121c2ffd0>),
                ('fill_na',
                 <imbd.preprocessing.FillNATransformer object at 0x121c2feb8>),
                ('variance_selector',
                 <imbd.preprocessing.VarianceFeatureSelector object at 0x10c092748>),
                ('outlier_detection',
                 <imbd.preprocessing.OutlierDetector object at 0x10b811898>)],
         verbose=True)
```

#### Featrue Importances

| Feature       |  Importance |
| :------------ | ----------: |
| Input_C_063_x |   0.0443957 |
| Input_C_071_x |   0.0399098 |
| Input_C_079_x |   0.0338355 |
| Input_C_082_y |   0.0291333 |
| Input_C_070_x |    0.025856 |
| Input_C_030_y |   0.0229639 |
| Input_C_022_y |   0.0226339 |
| Input_C_035_y |   0.0216159 |
| Input_C_026_x |   0.0211384 |
| Input_C_064_y |    0.018978 |
| Input_C_015_y |   0.0179436 |
| Input_C_022_x |   0.0178569 |
| Input_C_068_x |   0.0174463 |
| Input_C_017_x |   0.0163248 |
| Input_C_076_y |   0.0158087 |
| Input_C_025_x |   0.0147232 |
| Input_C_036_y |   0.0143646 |
| Input_C_031_x |   0.0139116 |
| Input_C_073_x |   0.0138195 |
| Input_A4_020  |   0.0134045 |
| Input_A3_020  |   0.0133536 |
| Input_C_072_x |   0.0131655 |
| Input_C_080_x |   0.0129474 |
| Input_C_081_x |    0.012902 |
| Input_C_074_y |   0.0127627 |
| Input_C_018_x |   0.0124593 |
| Input_C_020_x |   0.0124037 |
| Input_A3_001  |   0.0120893 |
| Input_C_066_x |   0.0117604 |
| Input_C_075_y |   0.0116633 |
| Input_C_070_y |   0.0113378 |
| Output_A4     |   0.0113321 |
| Input_C_016_x |   0.0111176 |
| Input_C_015_x |   0.0108532 |
| Input_C_019_y |   0.0107419 |
| Input_C_037_x |   0.0105389 |
| Input_A5_020  |   0.0104909 |
| Input_C_034_x |   0.0102285 |
| Input_C_018_y |  0.00994248 |
| Input_A6_020  |   0.0098133 |
| Input_A5_005  |  0.00979786 |
| Input_C_074_x |  0.00951218 |
| Output_A3     |   0.0094734 |
| Input_C_078_x |  0.00919017 |
| Input_C_019_x |  0.00917637 |
| Input_C_032_x |  0.00889124 |
| Input_C_038_x |   0.0081629 |
| Input_C_033_x |   0.0081473 |
| Input_C_027_x |  0.00796513 |
| Output_A1     |   0.0078221 |
| Input_C_082_x |   0.0078091 |
| Input_C_069_x |  0.00777241 |
| Output_A5     |  0.00766689 |
| Input_C_029_y |   0.0073299 |
| Input_C_026_y |  0.00727765 |
| Input_C_038_y |   0.0071133 |
| Output_A2     |  0.00705652 |
| Input_C_065_x |   0.0068919 |
| Input_C_076_x |  0.00684993 |
| Input_C_031_y |  0.00679604 |
| Input_C_135   |  0.00675162 |
| Input_C_028_y |  0.00668826 |
| Output_A6     |  0.00663834 |
| Input_A2_020  |  0.00649382 |
| Input_C_067_x |  0.00616471 |
| Input_C_020_y |  0.00610363 |
| Input_C_137   |  0.00588024 |
| Input_C_028_x |   0.0057809 |
| Input_C_063_y |  0.00570741 |
| Input_C_024_y |  0.00548147 |
| Input_C_017_y |  0.00546984 |
| Input_C_021_y |  0.00538265 |
| Input_C_034_y |  0.00509805 |
| Input_C_024_x |  0.00494541 |
| Input_C_025_y |  0.00480253 |
| Input_C_080_y |  0.00459626 |
| Input_C_066_y |  0.00458858 |
| Input_C_027_y |  0.00454957 |
| Input_C_023_x |  0.00443068 |
| Input_C_073_y |  0.00438381 |
| Input_C_036_x |  0.00425179 |
| Input_C_023_y |  0.00409878 |
| Input_C_021_x |  0.00409635 |
| Input_C_016_y |  0.00400374 |
| Input_C_078_y |  0.00391293 |
| Input_C_037_y |  0.00360936 |
| Input_C_075_x |  0.00354406 |
| Input_C_032_y |  0.00353395 |
| Input_C_071_y |  0.00349026 |
| Input_C_030_x |  0.00345208 |
| Input_C_035_x |  0.00339853 |
| Input_C_065_y |  0.00266325 |
| Input_C_069_y |   0.0024992 |
| Input_C_067_y |  0.00240972 |
| Input_C_079_y |  0.00236126 |
| Input_C_136   |  0.00233528 |
| Input_C_068_y |  0.00222432 |
| Input_C_072_y |  0.00192995 |
| Input_C_033_y |  0.00158577 |
| Input_C_029_x |  0.00135309 |
| Input_C_064_x | 0.000388244 |
| Input_C_077_x | 0.000169107 |
| Input_C_077_y | 8.11342e-05 |
| Input_C_081_y |           0 |
| outlier       |           0 |

#### Results

|      | mean_fit_time | std_fit_time | mean_score_time | std_score_time | param_estimator__max_depth | param_estimator__n_estimators | param_estimator__subsample | split0_test_score | split1_test_score | split2_test_score | mean_test_score | std_test_score | rank_test_score |
| ---: | ------------: | -----------: | --------------: | -------------: | -------------------------: | ----------------------------: | -------------------------: | ----------------: | ----------------: | ----------------: | --------------: | -------------: | --------------: |
|    1 |       9.02446 |     0.882759 |       0.0398846 |      0.0030827 |                          2 |                          1000 |                        0.5 |          0.164471 |          0.182591 |          0.115417 |         0.15416 |      0.0283761 |               1 |
|    0 |       8.63377 |     0.649166 |       0.0337801 |     0.00252528 |                          2 |                          1000 |                          1 |          0.151139 |          0.185689 |          0.109918 |        0.148915 |      0.0309732 |               2 |
|    4 |       9.49543 |      1.78376 |       0.0334764 |     0.00163987 |                         10 |                          1000 |                          1 |           0.15271 |          0.198517 |         0.0940489 |        0.148425 |      0.0427563 |               3 |
|    2 |       8.14894 |      1.14815 |       0.0319568 |     0.00282993 |                          5 |                          1000 |                          1 |          0.149009 |          0.188953 |         0.0972453 |        0.145069 |      0.0375431 |               4 |
|    3 |       7.63279 |     0.106976 |       0.0330773 |    0.000940062 |                          5 |                          1000 |                        0.5 |          0.145274 |          0.169932 |          0.101922 |        0.139043 |      0.0281125 |               5 |
|    5 |       7.77998 |      1.44659 |       0.0314119 |     0.00064167 |                         10 |                          1000 |                        0.5 |          0.148859 |          0.166153 |          0.100756 |         0.13859 |      0.0276681 |               6 |