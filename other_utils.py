def evaluate_model(model, data_stream):
    metric = utils.Rolling(MAE(), 12)

    y_trues = []
    y_preds = []

    for x, y in data_stream:
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        # Update the error metric
        metric.update(y, y_pred)

        # Store the true value and the prediction
        y_trues.append(y)
        y_preds.append(y_pred)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Ground truth')
    ax.plot(y_preds, lw=3, color='#e74c3c', alpha=0.8, label='Prediction')
    ax.legend()
    ax.set_title(str(metric))

# Make sure to include 'date' in your converters if it needs special parsing, e.g., to datetime
data_stream = stream.iter_csv(
    'data/air_pollution_dataset_modified.csv', 
    target='pred_pollution', 
    converters={
        'current_pollution': float, 
        'dew': float, 
        'temp': float, 
        'press': float, 
        'wnd_spd': float, 
        'snow': float, 
        'rain': float, 
        'pred_pollution': float,
        'date': pd.to_datetime  # Convert 'date' to datetime if it's not in a suitable format
    }
)

model = (compose.Select('wnd_dir') | preprocessing.OneHotEncoder())
model += (compose.Select('current_pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain') | preprocessing.StandardScaler())
model |= HoeffdingAdaptiveTreeRegressor(grace_period=250, drift_detector=drift.ADWIN())

evaluate_model(model, data_stream)


### Get metrics from iter_progressive_val_score

data_stream = stream.iter_csv('data/air_pollution_dataset_modified.csv', target='pred_pollution', converters={'current_pollution': float, 'dew': float, 'temp': float, 'press': float, 'wnd_spd': float, 'snow': float, 'rain': float, 'pred_pollution': float, 'date': pd.to_datetime})

# Define the pipeline
model = (compose.Select('wnd_dir') | preprocessing.OneHotEncoder())
model += (compose.Select('current_pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain')|preprocessing.StandardScaler())
model |=  HoeffdingAdaptiveTreeRegressor()

results = evaluate.iter_progressive_val_score(dataset=data_stream, model=model, metric=MAE(), step=500, yield_predictions=True)

import pandas as pd
import matplotlib.pyplot as plt

# Convert the generator to a list
results_list = list(results)

# Convert the list to a DataFrame
results_df = pd.DataFrame(results_list)


### Plot just one graph

def evaluate_model(model, data_stream):
    metric = MAE()

    dates = []
    y_trues = []
    y_preds = []

    for x, y in data_stream:
        # Store the date
        dates.append(x['date'])

        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)

        # Ensure the prediction is non-negative as pollution can't be negative (jeje)
        y_pred = max(0, y_pred)
        model.learn_one(x, y)

        # Update the error metric
        metric.update(y, y_pred)

        # Store the true value and the prediction
        y_trues.append(y)
        y_preds.append(y_pred)

    # Plot the results
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.grid(alpha=0.75)
    ax.plot(dates[40000:41000], y_trues[40000:41000], lw=1, color='#2ecc71', alpha=0.8, label='Ground truth')
    ax.plot(dates[40000:41000], y_preds[40000:41000], lw=1, color='#e74c3c', alpha=0.8, label='Prediction')
    ax.legend()
    ax.set_title(str(metric))
    plt.show()

    return metric