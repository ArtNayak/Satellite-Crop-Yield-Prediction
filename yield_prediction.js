// ============================================================
// SATELLITE-BASED YIELD PREDICTION - Google Earth Engine
// Sentinel-2 indices -> Random Forest regression -> yield (t/ha)
// Paste into https://code.earthengine.google.com and press Run.
// ============================================================

// -------------------------------
// 1. Configuration
// -------------------------------
var roi = ee.Geometry.Point([73.685, 20.015]).buffer(500); // vineyard, 500 m radius
Map.centerObject(roi, 16);

var TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023];
var TARGET_YEAR = 2024;
var SEASON_START = '-01-01';
var SEASON_END = '-06-01'; // end date is exclusive, so this covers Jan 1 - May 31

var S2_BANDS = ['B2', 'B3', 'B4', 'B8'];
var MODEL_BANDS = ['NDVI', 'EVI', 'NDWI', 'B2', 'B3', 'B4', 'B8'];
var SEED = 42;

// The satellite gives no yield numbers, so training labels come from a proxy
// score of each season's mean NDVI/EVI/NDWI, stretched onto this assumed
// yield range (t/ha). Replace the range with figures for your crop and region.
var REAL_MIN = 15;
var REAL_MAX = 30;

// Better: put your farm's measured yields here (t/ha per year). When set, they
// are used as the training labels and the proxy and range above are ignored.
// Example: {2019: 18.2, 2020: 21.5, 2021: 19.8, 2022: 24.1, 2023: 22.7}
var REAL_YIELDS = null;

// -------------------------------
// 2. Seasonal composite
// -------------------------------
// Harmonized collection: removes the +1000 offset Sentinel-2 added in 2022, so
// years are comparable. Reflectance is scaled to 0-1, which the EVI formula needs.
function maskAndScale(img) {
  var scl = img.select('SCL'); // scene classification
  var clear = scl.neq(3)    // cloud shadow
    .and(scl.neq(8))        // cloud, medium probability
    .and(scl.neq(9))        // cloud, high probability
    .and(scl.neq(10))       // thin cirrus
    .and(scl.neq(11));      // snow
  return img.updateMask(clear).select(S2_BANDS).divide(10000)
    .copyProperties(img, ['system:time_start']);
}

function getScenes(year) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(year + SEASON_START, year + SEASON_END)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));
}

function getSeasonImage(year) {
  return getScenes(year).map(maskAndScale).median().clip(roi);
}

// -------------------------------
// 3. Vegetation indices
// -------------------------------
function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      NIR: image.select('B8'),
      RED: image.select('B4'),
      BLUE: image.select('B2')
    }).rename('EVI');

  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');

  return image.addBands([ndvi, evi, ndwi]);
}

// -------------------------------
// 4. Proxy yield score (mean of the season's indices over the farm)
// -------------------------------
function proxyScore(image) {
  var s = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 10,
    maxPixels: 1e9
  });
  return ee.Number(s.get('NDVI')).multiply(2.5)
    .add(ee.Number(s.get('EVI')).multiply(1.5))
    .add(ee.Number(s.get('NDWI')))
    .multiply(2);
}

// Scenes per year: if a year has 0 scenes the composite is empty and the run fails.
TRAIN_YEARS.concat([TARGET_YEAR]).forEach(function(y) {
  print('Clear scenes in ' + y + ':', getScenes(y).size());
});

var seasonImages = {};
var proxies = [];
TRAIN_YEARS.forEach(function(y) {
  seasonImages[y] = addIndices(getSeasonImage(y));
  proxies.push(proxyScore(seasonImages[y]));
});

// Anchor the yield range to the training years (not to each year's own pixels),
// so a better year really does get a higher yield than a worse one.
var proxyList = ee.List(proxies);
var proxyMin = ee.Number(proxyList.reduce(ee.Reducer.min()));
var proxyMax = ee.Number(proxyList.reduce(ee.Reducer.max()));
var proxySpan = proxyMax.subtract(proxyMin).max(1e-6); // avoids dividing by zero

function labelFor(index) {
  if (REAL_YIELDS !== null) {
    return ee.Number(REAL_YIELDS[TRAIN_YEARS[index]]);
  }
  return ee.Number(proxies[index]).subtract(proxyMin).divide(proxySpan)
    .multiply(REAL_MAX - REAL_MIN).add(REAL_MIN);
}

// -------------------------------
// 5. Training samples (one set of pixels per year, labelled with that year's yield)
// -------------------------------
var sampleSets = TRAIN_YEARS.map(function(y, i) {
  var label = labelFor(i);
  return seasonImages[y].select(MODEL_BANDS).sample({
    region: roi,
    scale: 10,
    numPixels: 300,
    seed: SEED,
    geometries: true
  }).map(function(f) {
    return f.set({yield: label, year: y});
  });
});
var trainingSamples = ee.FeatureCollection(sampleSets).flatten();

print('Training samples:', trainingSamples.limit(5));

// -------------------------------
// 6. Random Forest
// -------------------------------
function makeRF() {
  return ee.Classifier.smileRandomForest({
    numberOfTrees: 200,
    variablesPerSplit: 3,
    minLeafPopulation: 5,
    seed: SEED
  }).setOutputMode('REGRESSION');
}

var trainedRF = makeRF().train({
  features: trainingSamples,
  classProperty: 'yield',
  inputProperties: MODEL_BANDS
});

// -------------------------------
// 7. Validation: leave one year out
// -------------------------------
// Every pixel of a year shares one label, so a random pixel split would leak:
// test pixels would sit next to identical training pixels from the same year.
// Instead each year is held out in turn and predicted by a model that never saw it.
var folds = TRAIN_YEARS.map(function(y) {
  var rfY = makeRF().train({
    features: trainingSamples.filter(ee.Filter.neq('year', y)),
    classProperty: 'yield',
    inputProperties: MODEL_BANDS
  });
  var held = trainingSamples.filter(ee.Filter.eq('year', y)).classify(rfY);
  var actual = ee.Number(held.first().get('yield'));
  var predicted = ee.Number(held.aggregate_mean('classification'));
  return ee.Feature(null, {
    year: y,
    actual_t_ha: actual,
    predicted_t_ha: predicted,
    abs_error: predicted.subtract(actual).abs()
  });
});
var foldTable = ee.FeatureCollection(folds);

print('Leave-one-year-out validation (t/ha):', foldTable);
print('Mean absolute error across years (t/ha):', foldTable.aggregate_mean('abs_error'));

// -------------------------------
// 8. Predict the target year (output is already in t/ha)
// -------------------------------
var targetImage = addIndices(getSeasonImage(TARGET_YEAR));
var yield_tpha = targetImage.select(MODEL_BANDS)
  .classify(trainedRF)
  .rename('Yield_ton_per_ha');

// -------------------------------
// 9. Farm production
// -------------------------------
var areaHa = roi.area().divide(10000);

var meanYield = yield_tpha.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 10,
  maxPixels: 1e9
}).get('Yield_ton_per_ha');

print('Farm area (ha):', areaHa);
print('Predicted mean yield ' + TARGET_YEAR + ' (t/ha):', meanYield);
print('Predicted total production (tons):', ee.Number(meanYield).multiply(areaHa));

// -------------------------------
// 10. Map layers
// -------------------------------
Map.addLayer(
  targetImage.select('NDVI'),
  {min: 0, max: 1, palette: ['brown', 'yellow', 'green']},
  'NDVI ' + TARGET_YEAR
);

Map.addLayer(
  yield_tpha,
  {min: REAL_MIN, max: REAL_MAX, palette: ['red', 'yellow', 'green']},
  'Predicted yield (t/ha)'
);

// -------------------------------
// 11. NDVI time series (training years + target year)
// -------------------------------
var allYears = TRAIN_YEARS.concat([TARGET_YEAR]);
var ndviSeries = allYears.map(function(y) {
  var img = (y === TARGET_YEAR) ? targetImage : seasonImages[y];
  return ee.Feature(null, {
    year: y,
    NDVI: img.select('NDVI').reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: roi,
      scale: 10,
      maxPixels: 1e9
    }).get('NDVI')
  });
});

print(ui.Chart.feature.byFeature({
  features: ee.FeatureCollection(ndviSeries),
  xProperty: 'year',
  yProperties: ['NDVI']
})
  .setChartType('LineChart')
  .setOptions({
    title: 'Mean NDVI per season',
    hAxis: {title: 'Year', format: '####'},
    vAxis: {title: 'Mean NDVI'},
    lineWidth: 3,
    pointSize: 6,
    colors: ['green']
  }));

// -------------------------------
// 12. Actual vs predicted yield per held-out year
// -------------------------------
print(ui.Chart.feature.byFeature({
  features: foldTable,
  xProperty: 'year',
  yProperties: ['actual_t_ha', 'predicted_t_ha']
})
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Leave-one-year-out: label vs prediction (t/ha)',
    hAxis: {title: 'Held-out year', format: '####'},
    vAxis: {title: 'Yield (t/ha)'}
  }));

// -------------------------------
// 13. Yield histogram
// -------------------------------
print(ui.Chart.image.histogram({
  image: yield_tpha,
  region: roi,
  scale: 10,
  maxPixels: 1e9
}).setOptions({
  title: 'Predicted yield distribution (t/ha)',
  hAxis: {title: 'Yield (t/ha)'},
  vAxis: {title: 'Pixel count'},
  colors: ['orange']
}));

// -------------------------------
// 14. Feature importance
// -------------------------------
var importance = ee.Dictionary(trainedRF.explain().get('importance'));
var importanceFC = ee.FeatureCollection(importance.keys().map(function(key) {
  return ee.Feature(null, {feature: key, importance: importance.get(key)});
}));

print(ui.Chart.feature.byFeature({
  features: importanceFC,
  xProperty: 'feature',
  yProperties: ['importance']
})
  .setChartType('ColumnChart')
  .setOptions({title: 'Random Forest feature importance', legend: {position: 'none'}}));

// -------------------------------
// 15. NDVI vs predicted yield (target year pixels)
// -------------------------------
var scatterSamples = yield_tpha.addBands(targetImage.select('NDVI')).sample({
  region: roi,
  scale: 10,
  numPixels: 500,
  seed: SEED,
  geometries: false
});

print(ui.Chart.feature.byFeature({
  features: scatterSamples,
  xProperty: 'NDVI',
  yProperties: ['Yield_ton_per_ha']
})
  .setChartType('ScatterChart')
  .setOptions({
    title: 'NDVI vs predicted yield',
    hAxis: {title: 'NDVI'},
    vAxis: {title: 'Yield (t/ha)'},
    pointSize: 3
  }));
