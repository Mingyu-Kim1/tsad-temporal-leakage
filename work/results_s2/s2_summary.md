# E0 부분표본 기울기 비교 (S2, gap_mult 1.0)

## adapter_latent  (중복 있는 pair 7개, 분리 가능 1개)

| 집합 | 평균 기울기 | median |
| --- | --- | --- |
| E0 | -0.2677 | -0.2349 |
| E2 | +0.0168 | +0.0193 |
| E0_matched | +0.0174 | +0.0195 |
| E0_mix | -0.0021 | -0.0021 |
| E2_mix | -0.0014 | -0.0014 |

- 분리 가능한 pair 가 1개라 검정하지 않음

## full_ft  (중복 있는 pair 7개, 분리 가능 1개)

| 집합 | 평균 기울기 | median |
| --- | --- | --- |
| E0 | -0.5589 | -0.5598 |
| E2 | -0.0176 | -0.0140 |
| E0_matched | -0.0175 | -0.0142 |
| E0_mix | -0.0058 | -0.0058 |
| E2_mix | -0.0064 | -0.0064 |

- 분리 가능한 pair 가 1개라 검정하지 않음

## pair 별

| pair           | method         |   n_contam |   overlap_frac |   mix_cov |   leak_share |   slope_E0 |   slope_E2 |   slope_E0_matched |   slope_E0_mix |   slope_E2_mix | separable   |
|:---------------|:---------------|-----------:|---------------:|----------:|-------------:|-----------:|-----------:|-------------------:|---------------:|---------------:|:------------|
| Boiler_1to2    | adapter_latent |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.2344 |    -0.0075 |            -0.0074 |       nan      |       nan      | False       |
| Boiler_1to2    | full_ft        |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.6228 |    -0.0259 |            -0.0259 |       nan      |       nan      | False       |
| Boiler_1to3    | adapter_latent |          5 |         0.2002 |    0.0009 |       0.0002 |    -0.641  |     0.0632 |             0.0632 |       nan      |       nan      | False       |
| Boiler_1to3    | full_ft        |          5 |         0.2002 |    0.0009 |       0.0002 |    -0.9269 |     0.0301 |             0.0302 |       nan      |       nan      | False       |
| Boiler_2to1    | adapter_latent |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.2958 |     0.0376 |             0.0376 |       nan      |       nan      | False       |
| Boiler_2to1    | full_ft        |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.4929 |     0.0209 |             0.0212 |       nan      |       nan      | False       |
| Boiler_2to3    | adapter_latent |          5 |         0.2002 |    0.0009 |       0.0002 |    -0.2733 |     0.0218 |             0.0218 |       nan      |       nan      | False       |
| Boiler_2to3    | full_ft        |          5 |         0.2002 |    0.0009 |       0.0002 |    -0.8109 |    -0.089  |            -0.089  |       nan      |       nan      | False       |
| Boiler_3to1    | adapter_latent |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.2349 |     0.0193 |             0.0195 |       nan      |       nan      | False       |
| Boiler_3to1    | full_ft        |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.4969 |    -0.0397 |            -0.039  |       nan      |       nan      | False       |
| Boiler_3to2    | adapter_latent |          5 |         0.8009 |    0.0011 |       0.0043 |    -0.1855 |    -0.0043 |            -0.0044 |       nan      |       nan      | False       |
| Boiler_3to2    | full_ft        |          1 |         0.8009 |    0.0011 |       0.0043 |   nan      |   nan      |           nan      |       nan      |       nan      | False       |
| MSL_G-6toE-1   | adapter_latent |          5 |         0.0947 |    1      |       0.0742 |    -0.0087 |    -0.0125 |            -0.0084 |        -0.0021 |        -0.0014 | True        |
| MSL_G-6toE-1   | full_ft        |          5 |         0.0947 |    1      |       0.0741 |    -0.0028 |    -0.0021 |            -0.0024 |        -0.0058 |        -0.0064 | True        |
| MSL_G-6toE-13  | adapter_latent |          5 |         0      |    0      |       0      |     0.0014 |     0.0014 |             0.0014 |       nan      |       nan      | False       |
| MSL_G-6toE-13  | full_ft        |          5 |         0      |    0      |       0      |    -0.0084 |    -0.0084 |            -0.0084 |       nan      |       nan      | False       |
| MSL_P-10toP-11 | adapter_latent |          5 |         0      |    0      |       0      |    -0      |    -0      |            -0      |       nan      |       nan      | False       |
| MSL_P-10toP-11 | full_ft        |          5 |         0      |    0      |       0      |    -0.0009 |    -0.0009 |            -0.0009 |       nan      |       nan      | False       |
| MSL_P-11toP-10 | adapter_latent |          5 |         0      |    0      |       0      |    -0.0001 |    -0.0001 |            -0.0001 |       nan      |       nan      | False       |
| MSL_P-11toP-10 | full_ft        |          5 |         0      |    0      |       0      |     0.0002 |     0.0002 |             0.0002 |       nan      |       nan      | False       |

separable 판정: 중복 window 중 비중복과 같은 10분위에 있는 비율(mix_cov) >= 0.5 이고 매칭 표본의 중복 비율(leak_share) >= 0.02.
separable 이 False 인 pair 는 중복이 특정 구간에 몰려 있어 누수와 구간 이동이 같은 위치에 겹친다. 이 경우 E0-matched 가 E2 에 가깝게 나와도 구간 이동의 증거로 해석하면 안 된다.

분리 불가 pair: Boiler_1to2, Boiler_1to3, Boiler_2to1, Boiler_2to3, Boiler_3to1, Boiler_3to2