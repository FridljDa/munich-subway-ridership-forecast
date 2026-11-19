# Problem statement: Munich subway ridership forecast
The data set in `data/` is synthetic.

## Business context

The goal is to forecast the number of passengers entering each subway station in Munich hourly (7am–12pm), 7 days in advance, in order to allocate operational resources to the right stations ahead of time.

## Dataset

You are given 4 tables:

- station_metadata
- timeseries_with_target
- traffic_hourly
- weather_hourly

## Task

Train a model to predict hourly passenger counts per station, 7 days ahead.

## Train / test split

- Train: all data except the last 7 days
- Test: last 7 days only (420 rows)

## Evaluation

Report:

- Overall RMSE on the test set
- RMSE per station on the test set

## Production assumptions

- station_metadata, timeseries_with_target are stored in an Azure Data Lake and updated hourly
- traffic_hourly and weather_hourly are accessed via APIs and updated hourly
- Accurate weather forecasts are available for the next 10 days

## Deliverable

Short executive summary covering:

- Problem framing
- Feature & model choice
- Evaluation results
- Production deployment & monitoring