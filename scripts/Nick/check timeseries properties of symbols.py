import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import plotly.express as px
import plotly.graph_objects as go
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gc

# Create output directory for plots
PLOT_DIR = Path("timeseries_analysis_plots")
PLOT_DIR.mkdir(exist_ok=True)

def analyze_symbol_timeseries(file_path):
    """Analyze a single symbol's timeseries data focusing on temporal patterns"""
    
    # Create main progress bar for this symbol
    main_pbar = tqdm(total=7, desc=f"Analyzing {file_path.name}", leave=True)
    
    # Data Loading
    main_pbar.set_description("Loading Data")
    df = pd.read_parquet(file_path, columns=['date_id', 'time_id'])
    symbol_id = file_path.stem.split('_')[1]
    main_pbar.update(1)
    
    # Basic Statistics
    main_pbar.set_description("Basic Statistics")
    stats_info = {
        'Total rows': len(df),
        'Memory usage': f"{df.memory_usage().sum() / 1024**2:.2f} MB"
    }
    main_pbar.update(1)
    
    # Temporal Analysis
    main_pbar.set_description("Temporal Analysis")
    temporal_info = {
        'Date range': f"{df['date_id'].min()} to {df['date_id'].max()}",
        'Time range': f"{df['time_id'].min()} to {df['time_id'].max()}",
        'Unique dates': df['date_id'].nunique(),
        'Unique times': df['time_id'].nunique()
    }
    main_pbar.update(1)
    
    # Data Processing
    main_pbar.set_description("Processing Data")
    df['date_id'] = df['date_id'].astype('int32')
    df['time_id'] = df['time_id'].astype('int32')
    
    # Create date-time hierarchical analysis
    date_time_counts = df.groupby(['date_id', 'time_id']).size().reset_index(name='count')
    date_counts = df.groupby('date_id').size()
    time_counts_by_date = df.groupby(['date_id', 'time_id']).size().unstack(fill_value=0)
    
    # Calculate statistics for each date's time distribution
    time_stats_by_date = pd.DataFrame({
        'date_id': date_counts.index,
        'total_records': date_counts.values,
        'unique_times': df.groupby('date_id')['time_id'].nunique(),
        'min_time': df.groupby('date_id')['time_id'].min(),
        'max_time': df.groupby('date_id')['time_id'].max(),
        'mean_records_per_time': df.groupby('date_id')['time_id'].size() / df.groupby('date_id')['time_id'].nunique()
    })
    main_pbar.update(1)
    
    # Plot Generation
    main_pbar.set_description("Generating Plots")
    symbol_dir = PLOT_DIR / f"symbol_{symbol_id}"
    symbol_dir.mkdir(exist_ok=True)
    
    plot_pbar = tqdm(total=7, desc="Plotting", leave=False)
    
    # 1. Daily Time Coverage Analysis
    plot_pbar.set_description("Time Coverage by Date")
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 16))
    
    # Heatmap of time coverage for last 50 days
    last_50_days = time_counts_by_date.iloc[-50:]
    sns.heatmap(last_50_days, cmap='YlOrRd', ax=axes1[0,0])
    axes1[0,0].set_title('Time Coverage Heatmap (Last 50 Days)')
    axes1[0,0].set_xlabel('Time ID')
    axes1[0,0].set_ylabel('Date ID')
    
    # Box plot of time distribution by date
    sns.boxplot(data=date_time_counts, x='date_id', y='time_id', 
                ax=axes1[0,1], showfliers=False)
    axes1[0,1].set_title('Time Distribution by Date')
    axes1[0,1].set_xlabel('Date ID')
    axes1[0,1].set_ylabel('Time ID')
    axes1[0,1].tick_params(axis='x', rotation=45)
    
    # Line plot of unique times per date
    sns.lineplot(data=time_stats_by_date, x='date_id', y='unique_times', ax=axes1[1,0])
    axes1[1,0].set_title('Number of Unique Times per Date')
    axes1[1,0].set_xlabel('Date ID')
    axes1[1,0].set_ylabel('Unique Times')
    
    # Histogram of mean records per time for each date
    sns.histplot(data=time_stats_by_date, x='mean_records_per_time', ax=axes1[1,1], kde=True)
    axes1[1,1].set_title('Distribution of Mean Records per Time across Dates')
    axes1[1,1].set_xlabel('Mean Records per Time')
    axes1[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig1.savefig(symbol_dir / 'daily_time_coverage.png')
    plt.close(fig1)
    plot_pbar.update(1)
    
    # 2. Time Pattern Analysis
    plot_pbar.set_description("Time Patterns")
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 16))
    
    # Average time pattern across all dates
    avg_time_pattern = time_counts_by_date.mean()
    sns.lineplot(x=avg_time_pattern.index, y=avg_time_pattern.values, ax=axes2[0,0])
    axes2[0,0].set_title('Average Time Pattern Across All Dates')
    axes2[0,0].set_xlabel('Time ID')
    axes2[0,0].set_ylabel('Average Records')
    
    # Time pattern variation
    time_pattern_std = time_counts_by_date.std()
    sns.lineplot(x=time_pattern_std.index, y=time_pattern_std.values, ax=axes2[0,1])
    axes2[0,1].set_title('Time Pattern Variation (Standard Deviation)')
    axes2[0,1].set_xlabel('Time ID')
    axes2[0,1].set_ylabel('Standard Deviation of Records')
    
    # Time coverage consistency
    time_coverage = (time_counts_by_date > 0).mean()
    sns.lineplot(x=time_coverage.index, y=time_coverage.values, ax=axes2[1,0])
    axes2[1,0].set_title('Time Coverage Consistency')
    axes2[1,0].set_xlabel('Time ID')
    axes2[1,0].set_ylabel('Proportion of Dates with Records')
    
    # Distribution of time gaps
    time_gaps = []
    for _, day_data in time_counts_by_date.iterrows():
        active_times = day_data[day_data > 0].index
        if len(active_times) > 1:
            gaps = np.diff(active_times)
            time_gaps.extend(gaps)
    
    if time_gaps:
        sns.histplot(time_gaps, ax=axes2[1,1], kde=True)
        axes2[1,1].set_title('Distribution of Time Gaps')
        axes2[1,1].set_xlabel('Gap Size (Time ID units)')
        axes2[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig2.savefig(symbol_dir / 'time_patterns.png')
    plt.close(fig2)
    plot_pbar.update(1)
    
    # 3. Time Consistency Analysis
    plot_pbar.set_description("Time Consistency")
    fig3, axes3 = plt.subplots(2, 2, figsize=(20, 16))
    
    # Time coverage completeness by date
    coverage_completeness = (time_counts_by_date > 0).sum(axis=1) / time_counts_by_date.shape[1]
    sns.lineplot(x=coverage_completeness.index, y=coverage_completeness.values, ax=axes3[0,0])
    axes3[0,0].set_title('Time Coverage Completeness by Date')
    axes3[0,0].set_xlabel('Date ID')
    axes3[0,0].set_ylabel('Proportion of Times Covered')
    
    # Distribution of records per time slot by date quartile
    date_quartiles = pd.qcut(date_time_counts['date_id'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    date_time_counts['date_quartile'] = date_quartiles
    sns.boxplot(data=date_time_counts, x='date_quartile', y='count', ax=axes3[0,1])
    axes3[0,1].set_title('Records per Time Slot by Date Quartile')
    
    # Time coverage pattern evolution
    rolling_coverage = pd.DataFrame(time_counts_by_date > 0).rolling(window=7).mean()
    sns.heatmap(rolling_coverage.T, cmap='YlOrRd', ax=axes3[1,0])
    axes3[1,0].set_title('7-Day Rolling Time Coverage Pattern')
    axes3[1,0].set_xlabel('Date ID')
    axes3[1,0].set_ylabel('Time ID')
    
    # Daily time range
    daily_time_range = time_stats_by_date['max_time'] - time_stats_by_date['min_time']
    sns.histplot(daily_time_range, ax=axes3[1,1], kde=True)
    axes3[1,1].set_title('Distribution of Daily Time Range')
    axes3[1,1].set_xlabel('Time Range (max - min)')
    axes3[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig3.savefig(symbol_dir / 'time_consistency.png')
    plt.close(fig3)
    plot_pbar.update(1)
    
    # 4. Time Density Analysis
    plot_pbar.set_description("Time Density")
    fig4, axes4 = plt.subplots(2, 2, figsize=(20, 16))
    
    # Violin plot of time distribution by date quartiles
    sns.violinplot(data=date_time_counts, x='date_quartile', y='time_id', ax=axes4[0,0])
    axes4[0,0].set_title('Time Distribution Density by Date Quartiles')
    axes4[0,0].set_xlabel('Date Quartile')
    axes4[0,0].set_ylabel('Time ID')
    
    # Time gap analysis by date
    time_gaps_by_date = []
    for date_id in df['date_id'].unique():
        date_times = sorted(df[df['date_id'] == date_id]['time_id'].unique())
        if len(date_times) > 1:
            gaps = np.diff(date_times)
            time_gaps_by_date.extend([(date_id, gap) for gap in gaps])
    
    gaps_df = pd.DataFrame(time_gaps_by_date, columns=['date_id', 'gap'])
    sns.scatterplot(data=gaps_df, x='date_id', y='gap', alpha=0.5, ax=axes4[0,1])
    axes4[0,1].set_title('Time Gaps by Date')
    axes4[0,1].set_xlabel('Date ID')
    axes4[0,1].set_ylabel('Gap Size (Time ID units)')
    
    # Distribution of active time periods per date
    active_periods = []
    for date_id in df['date_id'].unique():
        date_times = sorted(df[df['date_id'] == date_id]['time_id'].unique())
        gaps = np.diff(date_times)
        large_gaps = np.where(gaps > 60)[0]  # Gap > 60 time units indicates new period
        active_periods.append(len(large_gaps) + 1)
    
    sns.histplot(active_periods, kde=True, ax=axes4[1,0])
    axes4[1,0].set_title('Distribution of Active Periods per Date')
    axes4[1,0].set_xlabel('Number of Active Periods')
    axes4[1,0].set_ylabel('Frequency')
    
    # Time density across all dates
    all_time_density = df.groupby('time_id').size()
    sns.lineplot(x=all_time_density.index, y=all_time_density.values, ax=axes4[1,1])
    axes4[1,1].set_title('Overall Time ID Density')
    axes4[1,1].set_xlabel('Time ID')
    axes4[1,1].set_ylabel('Number of Records')
    
    plt.tight_layout()
    fig4.savefig(symbol_dir / f'time_density_{symbol_id}.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    plot_pbar.update(1)
    
    # Missing Values Analysis
    plot_pbar.set_description("Missing Values")
    fig_missing, axes_missing = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Missing values heatmap over time
    missing_by_time = time_counts_by_date == 0
    sns.heatmap(missing_by_time.iloc[-50:], cmap='YlOrRd', ax=axes_missing[0,0])
    axes_missing[0,0].set_title('Missing Values Heatmap (Last 50 Days)')
    axes_missing[0,0].set_xlabel('Time ID')
    axes_missing[0,0].set_ylabel('Date ID')
    
    # 2. Missing values percentage by date
    missing_pct_by_date = (missing_by_time.sum(axis=1) / missing_by_time.shape[1]) * 100
    sns.lineplot(x=missing_pct_by_date.index, y=missing_pct_by_date.values, ax=axes_missing[0,1])
    axes_missing[0,1].set_title('Percentage of Missing Values by Date')
    axes_missing[0,1].set_xlabel('Date ID')
    axes_missing[0,1].set_ylabel('Missing Values (%)')
    
    # 3. Missing values percentage by time
    missing_pct_by_time = (missing_by_time.sum(axis=0) / missing_by_time.shape[0]) * 100
    sns.lineplot(x=missing_pct_by_time.index, y=missing_pct_by_time.values, ax=axes_missing[1,0])
    axes_missing[1,0].set_title('Percentage of Missing Values by Time ID')
    axes_missing[1,0].set_xlabel('Time ID')
    axes_missing[1,0].set_ylabel('Missing Values (%)')
    
    # 4. Distribution of consecutive missing values
    consecutive_missing = []
    for _, row in missing_by_time.iterrows():
        missing_streak = 0
        for val in row:
            if val:
                missing_streak += 1
            elif missing_streak > 0:
                consecutive_missing.append(missing_streak)
                missing_streak = 0
        if missing_streak > 0:
            consecutive_missing.append(missing_streak)
    
    if consecutive_missing:
        sns.histplot(consecutive_missing, kde=True, ax=axes_missing[1,1])
        axes_missing[1,1].set_title('Distribution of Consecutive Missing Values')
        axes_missing[1,1].set_xlabel('Length of Consecutive Missing Values')
        axes_missing[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    fig_missing.savefig(symbol_dir / f'missing_values_{symbol_id}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_missing)
    plot_pbar.update(1)
    
    # 5. Time Sequence Analysis
    plot_pbar.set_description("Time Sequences")
    fig5, axes5 = plt.subplots(2, 2, figsize=(20, 16))
    
    # First and last time_id by date
    first_last_times = pd.DataFrame({
        'date_id': df['date_id'].unique(),
        'first_time': df.groupby('date_id')['time_id'].min(),
        'last_time': df.groupby('date_id')['time_id'].max()
    })
    
    sns.scatterplot(data=first_last_times, x='date_id', y='first_time', 
                    label='First Time', ax=axes5[0,0], alpha=0.5)
    sns.scatterplot(data=first_last_times, x='date_id', y='last_time', 
                    label='Last Time', ax=axes5[0,0], alpha=0.5)
    axes5[0,0].set_title('First and Last Time ID by Date')
    axes5[0,0].set_xlabel('Date ID')
    axes5[0,0].set_ylabel('Time ID')
    
    # Time coverage duration by date
    first_last_times['duration'] = first_last_times['last_time'] - first_last_times['first_time']
    sns.histplot(data=first_last_times, x='duration', kde=True, ax=axes5[0,1])
    axes5[0,1].set_title('Distribution of Daily Time Coverage Duration')
    axes5[0,1].set_xlabel('Duration (Time ID units)')
    axes5[0,1].set_ylabel('Frequency')
    
    # Rolling average of time coverage
    window_size = 7
    rolling_first = first_last_times['first_time'].rolling(window=window_size).mean()
    rolling_last = first_last_times['last_time'].rolling(window=window_size).mean()
    
    sns.lineplot(x=first_last_times.index, y=rolling_first, 
                label=f'{window_size}-day Rolling Avg First Time', ax=axes5[1,0])
    sns.lineplot(x=first_last_times.index, y=rolling_last, 
                label=f'{window_size}-day Rolling Avg Last Time', ax=axes5[1,0])
    axes5[1,0].set_title('Rolling Average of Daily Time Coverage')
    axes5[1,0].set_xlabel('Date ID')
    axes5[1,0].set_ylabel('Time ID')
    
    # Time consistency score by date
    expected_times = set(range(int(df['time_id'].min()), int(df['time_id'].max()) + 1))
    consistency_scores = []
    
    for date_id in df['date_id'].unique():
        date_times = set(df[df['date_id'] == date_id]['time_id'].unique())
        active_range = set(range(min(date_times), max(date_times) + 1))
        expected_active = expected_times.intersection(active_range)
        score = len(date_times) / len(expected_active) if expected_active else 0
        consistency_scores.append((date_id, score))
    
    consistency_df = pd.DataFrame(consistency_scores, columns=['date_id', 'score'])
    sns.lineplot(data=consistency_df, x='date_id', y='score', ax=axes5[1,1])
    axes5[1,1].set_title('Time Coverage Consistency Score by Date')
    axes5[1,1].set_xlabel('Date ID')
    axes5[1,1].set_ylabel('Consistency Score')
    
    plt.tight_layout()
    fig5.savefig(symbol_dir / f'time_sequence_{symbol_id}.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    plot_pbar.update(1)
    
    # 6. Intraday Patterns
    plot_pbar.set_description("Intraday Patterns")
    fig6, axes6 = plt.subplots(2, 2, figsize=(20, 16))
    
    # Average activity by time_id
    time_activity = df.groupby('time_id').size()
    sns.lineplot(x=time_activity.index, y=time_activity.values, ax=axes6[0,0])
    axes6[0,0].set_title('Average Activity by Time ID')
    axes6[0,0].set_xlabel('Time ID')
    axes6[0,0].set_ylabel('Number of Records')
    
    # Activity distribution by time periods
    df['time_period'] = pd.qcut(df['time_id'], 
                              q=4, 
                              labels=['Period 1', 'Period 2', 'Period 3', 'Period 4'])
    sns.boxplot(data=df, x='time_period', y='time_id', ax=axes6[0,1])
    axes6[0,1].set_title('Activity Distribution by Time Period')
    
    # Time pattern stability
    time_patterns = df.groupby(['date_id', 'time_id']).size().unstack(fill_value=0)
    time_stability = time_patterns.std() / time_patterns.mean()
    sns.lineplot(x=time_stability.index, y=time_stability.values, ax=axes6[1,0])
    axes6[1,0].set_title('Time Pattern Stability (Lower = More Stable)')
    axes6[1,0].set_xlabel('Time ID')
    axes6[1,0].set_ylabel('Coefficient of Variation')
    
    # Intraday time gap distribution
    sns.boxplot(data=gaps_df, x=pd.qcut(gaps_df['date_id'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']), 
                y='gap', ax=axes6[1,1])
    axes6[1,1].set_title('Time Gap Distribution by Date Quartile')
    axes6[1,1].set_xlabel('Date Quartile')
    axes6[1,1].set_ylabel('Gap Size (Time ID units)')
    
    plt.tight_layout()
    fig6.savefig(symbol_dir / f'intraday_patterns_{symbol_id}.png', dpi=300, bbox_inches='tight')
    plt.close(fig6)
    plot_pbar.update(1)
    
    plot_pbar.close()
    main_pbar.update(1)
    
    # Stats Collection
    main_pbar.set_description("Collecting Stats")
    stats = {
        'symbol_id': symbol_id,
        'n_rows': len(df),
        # ... rest of stats collection ...
    }
    main_pbar.update(1)
    
    # Cleanup
    main_pbar.set_description("Cleanup")
    plt.close('all')
    del df
    gc.collect()
    main_pbar.update(1)
    
    main_pbar.close()
    return stats

def main():
    data_dir = Path(r'E:\coding projects\2024\jane street 2024\jane-street-real-time-market-data-forecasting\merged_symbols_dataset')
    parquet_files = list(data_dir.glob('*.parquet'))
    
    if not parquet_files:
        print("No parquet files found in the specified directory")
        return
    
    print(f"\nFound {len(parquet_files)} parquet files")
    print(f"Output directory: {PLOT_DIR}")
    
    # Reduce batch size
    BATCH_SIZE = 2  # Reduced from 4 to 2
    n_cores = min(2, max(1, multiprocessing.cpu_count() - 1))  # Limit to 2 cores
    
    # Add overall progress bar
    overall_progress = tqdm(total=len(parquet_files), desc="Overall Progress")
    
    results = []
    for i in range(0, len(parquet_files), BATCH_SIZE):
        batch_files = parquet_files[i:i + BATCH_SIZE]
        batch_progress = tqdm(desc=f"Batch {i//BATCH_SIZE + 1}", leave=False)
        
        # Process batch in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            future_to_file = {executor.submit(analyze_symbol_timeseries, file_path): file_path 
                            for file_path in batch_files}
            
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    overall_progress.update(1)
                    batch_progress.update(1)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {str(e)}")
        
        batch_progress.close()
        gc.collect()
    
    overall_progress.close()
    
    # Create and display summary report
    summary_df = pd.DataFrame(results)
    print("\nAnalysis Summary:")
    print(summary_df.describe())
    summary_df.to_csv(PLOT_DIR / "analysis_summary.csv", index=False)
    print(f"\nAnalysis complete. Results saved to {PLOT_DIR}")

if __name__ == "__main__":
    # This guard is important for parallel processing on Windows
    main()
