import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
from data_lib.variables import FEAT_NAMES, RESP_NAMES
from io_lib.paths import FIGS_DIR
from plot_lib.features import plot_feature_by_symbols

# Todo: Read from file
data_ixd = None
figs_dir = FIGS_DIR

# name_feature = 'feature_00'
for name_feature in FEAT_NAMES:
    print(f'Plotting {name_feature}')
    f, axs = plot_feature_by_symbols(data_ixd, name_feat=name_feature,
                                     figs_dir=figs_dir)

    data_tmp = data_ixd[name_feature].unstack().T
    data_tmp.corr()

    f, ax = plt.subplots()
    sns.heatmap(data_tmp.corr(), ax=ax)
    ax.set(title=name_feature.title())
    f.savefig(figs_dir / f'{name_feature}_corr.png')
    plt.close(f)


for name_resp in RESP_NAMES:
    print(f'Plotting {name_resp}')
    f, axs = plot_feature_by_symbols(data_ixd, name_feat=name_resp,
                                     figs_dir=figs_dir)

    data_tmp = data_ixd[name_resp].unstack().T
    data_tmp.corr()

    f, ax = plt.subplots()
    sns.heatmap(data_tmp.corr(), ax=ax)
    ax.set(title=name_resp.title())
    f.savefig(figs_dir / f'{name_resp}_corr.png')
    plt.close(f)



# Looking at lag
sym = 1
ds = data_ixd.loc[sym, 'responder_6']
ds_lag = ds.shift(-1)
df = pd.concat([ds, ds_lag], axis=1,
               keys=['lag', 'target']).dropna()

f, ax = plt.subplots()
ax.scatter(df['lag'], df['target'], alpha=0.01)
ax.set(title='Responder vs Its Lag', xlabel='lag 6',
       ylabel='responder 6')
f.savefig(figs_dir / f'{sym}_lag.png')
plt.close(f)
print(df.corr())

ds_corr = pd.Series(index=data_ixd.columns)
x = data_ixd.loc[sym, 'responder_6']
for c in data_ixd.columns:
    if c not in ['date_id', 'time_id', 'responder_6']:
        print(f'Calculating {c}')
        y = data_ixd.loc[sym, c]
        df_tmp = pd.concat((x, y), axis=1)
        ds_corr[c] = df_tmp.corr().iloc[0, 1]

ds_corr.dropna(inplace=True)

f, ax = plt.subplots(figsize=(20, 8))
ds_corr.plot.bar(ax=ax)
f.savefig(figs_dir / f'{sym}_correlations.png')
f.tight_layout()
plt.close(f)

