
DATES = None
TIMES = None
SYMBOLS = list(range(39))
RESPONDERS = list(range(9))


FEAT_NAMES = [f'feature_0{i}' for i in range(10)] + \
    [f'feature_{i}' for i in range(10, 79)]

RESP_NAMES = [f'responder_{i}' for i in range(9)]

FEATS = [f"feature_{i:02d}" for i in range(79)]
FEATS_TIME_LAG = [f"feature_{i:02d}_lag_1" for i in range(79)]
RESP = [f"responder_{i}" for i in range(9)]
RESP_DAY_LAG = [f"responder_{i}_lag_1" for i in range(9)]

FEATS_TOP_50 = ['feature_06', 'feature_36', 'feature_76',
                'feature_17', 'feature_22', 'feature_61',
                'feature_09', 'feature_10', 'feature_24',
                'feature_21', 'feature_59', 'symbol_id',
                'feature_30', 'time_id', 'feature_23',
                'feature_25', 'feature_15', 'feature_75',
                'feature_27', 'feature_38', 'feature_20',
                'feature_07', 'feature_34', 'feature_11',
                'feature_26', 'feature_35', 'feature_31',
                'feature_08', 'feature_66', 'feature_04',
                'feature_29',
                'feature_28', 'feature_52', 'feature_32',
                'feature_58', 'feature_78', 'feature_50',
                'feature_01',
                'feature_74', 'feature_70', 'feature_37',
                'feature_73', 'feature_05', 'feature_33',
                'feature_68',
                'feature_62', 'feature_02', 'feature_72',
                'feature_69', 'feature_14']


TARGET = 'responder_6'