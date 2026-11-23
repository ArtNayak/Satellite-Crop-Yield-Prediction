# Satellite-Based Crop Yield Prediction

A Google Earth Engine script that estimates the yield of a vineyard (default: a 500 m radius around 20.015°N, 73.685°E) from Sentinel-2 imagery. It builds a seasonal composite for each year, computes vegetation indices, trains a Random Forest regression on 2019–2023 and predicts the yield map for 2024, in tons per hectare.

## How it works
1. **Seasonal composite**: Sentinel-2 surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`) for Jan 1 – May 31 of each year. Cloud, shadow, cirrus and snow pixels are masked with the scene classification band, and the median is taken.
2. **Indices**: NDVI, EVI and NDWI added to the four bands B2, B3, B4 and B8.
3. **Training labels**: one yield value per training year (see *Labels* below).
4. **Model**: Random Forest (200 trees) on 300 sampled pixels per year, using 7 features.
5. **Validation**: leave-one-year-out. Each year is predicted by a model trained on the other four.
6. **Outputs**: yield map in t/ha, farm area, mean yield and total production, NDVI and yield maps, plus charts (NDVI trend, validation, yield histogram, feature importance, NDVI vs yield).

## Run
1. Open the [Earth Engine Code Editor](https://code.earthengine.google.com) (a free Earth Engine account is required).
2. Paste `yield_prediction.js` into a new script and press **Run**.
3. To use your own farm, change `roi` and the year settings at the top of the file.

## Labels: read this before trusting the numbers
Satellite images contain no yield measurements, so by default the labels are a **proxy**. Each year's mean NDVI, EVI and NDWI are combined into a score, which is stretched linearly onto an assumed yield range (`REAL_MIN` = 15 to `REAL_MAX` = 30 t/ha). The least-productive training year therefore gets 15 and the most productive gets 30, whatever the real yields were. The model learns to reproduce this index-based score, so the output ranks years and pixels by vegetation vigour. It is **not** a measured yield forecast.

For real predictions, set `REAL_YIELDS` near the top of the script to your farm's measured yields per year, for example `{2019: 18.2, 2020: 21.5, ...}`. These are then used as the labels and the proxy is ignored. With only five training years the model is still very small, so more years of measured data will help most.

## Changes from the original script
- **Harmonized Sentinel-2 collection**: the original `S2_SR` collection has a +1000 reflectance offset in scenes after January 2022, which made 2022–2024 incomparable with earlier years.
- **Reflectance scaled to 0–1**: the EVI formula (with its `+1` and `7.5` constants) needs this. Before, EVI was computed on 0–10000 values and was wrong.
- **Pixel-level cloud masking** instead of only skipping cloudy scenes.
- **Leave-one-year-out validation** replaces the random 70/30 pixel split. All pixels of a year share one label, so the old split put near-identical pixels in both sets and the RMSE was far too optimistic.
- **Yield calibration anchored to the training years.** The original stretched each prediction map's own min and max onto 15–30, so any year's best pixel was always 30. Now a better year really does get a higher yield, and the target year can fall outside the training range.
- **Optional real yields** (`REAL_YIELDS`), fixed random seeds, clear-scene counts per year to catch empty seasons, and all settings gathered at the top.

## Testing status
The revised script has not yet been run end to end in the Earth Engine Code Editor. It was syntax-checked and its client-side logic was exercised against a stub only. If a step errors in the Code Editor, check the printed scene counts first: a year with 0 clear scenes (a larger `CLOUDY_PIXEL_PERCENTAGE` limit or a longer season helps) is the most likely cause.
