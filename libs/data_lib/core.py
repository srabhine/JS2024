import os

def create_symbol_datasets(df,
                           base_path="C:/Users/srabh/Downloads/janestreettest"):
    """
    Creates separate datasets for each financial product (symbol_id),
    with all time series data in a single CSV file.

    Args:
        df: pandas DataFrame containing the data
        base_path: base directory where datasets will be stored
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Get unique symbol_ids
    symbol_ids = df['symbol_id'].unique()

    # Process each symbol_id (financial product)
    for symbol_id in symbol_ids:
        # Create directory for this symbol
        symbol_dir = os.path.join(base_path, str(symbol_id))
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)
        # Get all data for this symbol and sort by time
        symbol_data = df[df['symbol_id'] == symbol_id].sort_values(
            'time_id')

        # Save all data for this symbol in a single file
        file_path = os.path.join(symbol_dir, str(symbol_id) + '.csv')
        symbol_data.to_csv(file_path, index=False)

        print(
            f"Created dataset for symbol_id {symbol_id} with {len(symbol_data)} rows")