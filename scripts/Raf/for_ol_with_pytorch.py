"""

@author: Raffaele M Ghigliazza
"""


# while True:
#     inputs = last_targets
#     targets = get_latest_sample()
#     outputs, hidden = model(inputs, hidden)
#
#     optimizer.zero_grad()
#     loss = criterion(outputs, targets)
#     loss.backward()
#     optimizer.step()
#
#     last_targets = targets


global list_test, cnt, cnt_day
list_test = []
cnt_day = 0


def predict(test: pl.DataFrame,
            lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:

    global list_test, lags_, model, cnt, cnt_day

    if lags is not None:
        lags_ = lags
        lags_ = lags_.to_pandas()

    test_df = test.to_pandas().fillna(0.0)

    feature_names = [f"feature_{i:02d}" for i in range(79)]

    # print(f"\r{test_df['time_id'].iloc[0]}", end='')

    if test_df['time_id'].iloc[0] == 0:
        cnt_day += 1

    # print(test_df['date_id'].iloc[0])
    if cnt_day > 1:
        if test_df['time_id'].iloc[0] == 0:
            test_all = pd.concat(list_test, axis=0)
            test_all = pd.merge(test_all, lags_, on=["symbol_id","time_id"],
                                suffixes=("", "b"))
            test_all = test_all.fillna(0)
            # for i, model in enumerate(models):
            #     model.fit(x=test_all[feature_names],
            #     y=test_all['responder_6_lag_1'], sample_weight=test_all['weight'],
            #         batch_size=32, epochs=2,verbose=1,  shuffle=False)

            for i, model in enumerate(models):
                outputs = model(test_all[feature_names])

                optimizer.zero_grad()
                loss = criterion(outputs, test_all['responder_6_lag_1'])
                loss.backward()
                optimizer.step()

            list_test = [test_df]

        else:
            list_test.append(test_df)

    else:
        list_test.append(test_df)

    preds = np.zeros((test_df.shape[0],))
    for md in models:
        preds += md.predict(test_df[feature_names],
                            verbose=0).ravel() / len(models)

    predictions = \
        test.select('row_id'). \
            with_columns(
            pl.Series(
                name='responder_6',
                values=np.clip(preds, a_min=-5, a_max=5),
                dtype=pl.Float64,
            )
        )

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert list(predictions.columns) == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions
