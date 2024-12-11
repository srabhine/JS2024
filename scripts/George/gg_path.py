
is_linux = True
if is_linux:
    original_data_path = f"/home/zt/pyProjects/Optiver/JaneStreetMktPred/data/jane-street-real-time-market-data-forecasting/train.parquet"
    model_saving_path = "/home/zt/pyProjects/JaneSt/Team/scripts/George/6_0_ML_CAT/models"
    train_data_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD2"
    valid_data_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/validation.parquet"
    feature_dict_path = "/home/zt/pyProjects/JaneSt/Team/data/features_types.csv"

else:
    path = f"E:\Python_Projects\Optiver\JaneStreetMktPred\data\jane-street-real-time-market-data-forecasting\\train.parquet"
    merged_scaler_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\merged_scalers_df.pkl'
    scaler_std_df_path = 'E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\0_1_Transform_and_save_Data\\temp_save\scaler_std_df.pkl'
    feature_dict_path = "E:\Python_Projects\JS2024\GITHUB_C\data\\features_types.csv"
    model_saving_path = "E:\Python_Projects\JS2024\GITHUB_C\scripts\George\\1_0_NN_PlainVanilla\model_save\model_6_perSymbol_scale"
    train_data_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/FOLD2"
    valid_data_path = "/home/zt/pyProjects/Optiver/JaneStreetMktPred/Lag_XGB/data/validation.parquet"