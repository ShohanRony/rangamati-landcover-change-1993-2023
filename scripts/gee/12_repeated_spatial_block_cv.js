// ============================================================================
// Rangamati Land-Cover Change, 1993-2023
// Script 12 - Repeated Spatial-Block Cross-Validation + Per-Sample Predictions
// Researcher: Shohinur Pervez Shohan, RMSTU
// ----------------------------------------------------------------------------
// STATUS: production (added for the ICCIT 2026 camera-ready revision,
//   addressing Reviewer #4 comment 5: single holdout, no significance test).
//
// Purpose:  Re-evaluate RF / SVM / CART on EXACTLY the same 750-point,
//           0.1-degree-blocked sample as Script 04, but with
//             (a) the original seed-42 single holdout (rep = 0), which must
//                 reproduce Script 04's 77.10 / 78.04 / 72.90 % (sanity check),
//             (b) 5-fold spatial-block cross-validation repeated with 5
//                 different block-to-fold seeds (rep = seed).
//           Whole 0.1-degree blocks are assigned to folds, so no block is
//           ever split between training and testing.
//           Every test point's prediction from all three classifiers is
//           exported, so that summary statistics, kappa and McNemar's paired
//           test are computed offline in Python (Script 13).
// Inputs:   same as Script 04.
// Outputs:  Drive CSV  Script12_SpatialCV_Predictions.csv
//           columns: rep, fold, block_id, landcover, p_rf, p_svm, p_cart
// Depends:  none (self-contained; regenerates Script 04's sample).
// Params:   sampling/holdout seed = 42 (unchanged); CV seeds below; K = 5;
//           classifier hyperparameters identical to Script 04.
// ============================================================================

var studyArea = ee.FeatureCollection('projects/crypto-hallway-405211/assets/BGD_adm2')
  .filter(ee.Filter.eq('NAME_2', 'Parbattya Chattagram'));
// Map.centerObject(studyArea, 9);
// Map.addLayer(studyArea, {color: 'white'}, 'Study area');

// ---------------------------------------------------------------------------
// Landsat preprocessing
// ---------------------------------------------------------------------------
function maskAndScale(img) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0)
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0));
  var saturated = img.select('QA_RADSAT').eq(0);
  var optical = img.select('SR_B.*').multiply(0.0000275).add(-0.2);
  return img.addBands(optical, null, true)
    .updateMask(clear).updateMask(saturated)
    .copyProperties(img, img.propertyNames());
}

// Roy et al. (2016): OLI -> ETM+ harmonization (Landsat 8 only; Landsat 5
// and 7 already share the same band definitions, so they are used natively).
function harmonizeL8(img) {
  var slopes     = ee.Image.constant([0.8850,0.9317,0.9372,0.8339,0.8639,0.9165]);
  var intercepts = ee.Image.constant([0.0183,0.0123,0.0123,0.0448,0.0306,0.0116]);
  var l8  = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']);
  var out = l8.multiply(slopes).add(intercepts)
              .rename(['Blue','Green','Red','NIR','SWIR1','SWIR2']);
  return out.copyProperties(img, img.propertyNames());
}

function harmonizeL5(img) {
  var optical = img.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']);
  return optical.rename(['Blue','Green','Red','NIR','SWIR1','SWIR2'])
    .copyProperties(img, img.propertyNames());
}

function harmonizeL7(img) {
  var optical = img.select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']);
  return optical.rename(['Blue','Green','Red','NIR','SWIR1','SWIR2'])
    .copyProperties(img, img.propertyNames());
}

function addIndices(img) {
  var ndvi = img.normalizedDifference(['NIR','Red']).rename('NDVI');
  var ndwi = img.normalizedDifference(['Green','NIR']).rename('NDWI');
  var evi  = img.expression(
    '2.5*((nir-red)/(nir+6*red-7.5*blue+1))',
    {nir:img.select('NIR'),red:img.select('Red'),blue:img.select('Blue')}
  ).rename('EVI');
  var nbr  = img.normalizedDifference(['NIR','SWIR2']).rename('NBR');
  return img.addBands([ndvi, ndwi, evi, nbr]);
}

// Compositing window: January 1 - April 30 (dry season), no separate
// scene-level CLOUD_COVER pre-filter. Cloud/shadow contamination is handled
// entirely at the pixel level via QA_PIXEL + QA_RADSAT masking (maskAndScale,
// above) combined with median compositing across the window. One sensor per
// epoch — matching the sensor-by-year assignment used in Scripts 05/07/09
// (Landsat 5 for 1993/1998/2008, Landsat 7 for 2003, Landsat 8 for
// 2018/2023) — so the classifier comparison below is trained on the same
// kind of composite that produces the final production maps.
var epochs = [
  {year: 1993, col: 'LANDSAT/LT05/C02/T1_L2', harmonize: harmonizeL5},
  {year: 1998, col: 'LANDSAT/LT05/C02/T1_L2', harmonize: harmonizeL5},
  {year: 2003, col: 'LANDSAT/LE07/C02/T1_L2', harmonize: harmonizeL7},
  {year: 2008, col: 'LANDSAT/LT05/C02/T1_L2', harmonize: harmonizeL5},
  {year: 2018, col: 'LANDSAT/LC08/C02/T1_L2', harmonize: harmonizeL8},
  {year: 2023, col: 'LANDSAT/LC08/C02/T1_L2', harmonize: harmonizeL8}
];

function makeComposite(ep) {
  return ee.ImageCollection(ep.col)
    .filterBounds(studyArea)
    .filterDate(ep.year + '-01-01', ep.year + '-04-30')
    .map(maskAndScale)
    .map(ep.harmonize)
    .map(addIndices)
    .median().clip(studyArea).set('year', ep.year);
}

var srtm    = ee.Image('USGS/SRTMGL1_003').clip(studyArea);
var terrain = srtm.rename('elevation')
                  .addBands(ee.Terrain.slope(srtm).rename('slope'));
function withTerrain(img) { return img.addBands(terrain); }

var composite2023 = withTerrain(makeComposite(epochs[5]));

var featureBands = ['Blue','Green','Red','NIR','SWIR1','SWIR2',
                    'NDVI','NDWI','EVI','NBR','elevation','slope'];

// ---------------------------------------------------------------------------
// Class map — 5 classes
// 1=Dense Forest  2=Degraded Forest/Shrub/Grass  3=Water
// 4=Agriculture/Settlement  5=Bare Land
// ---------------------------------------------------------------------------
// Explicit versioned ESA WorldCover asset id, so the exact source dataset
// version is unambiguous and citable in the paper.
var worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(studyArea);
var ndvi2023   = composite2023.select('NDVI');
var tree       = worldcover.eq(10);

var dense    = tree.and(ndvi2023.gte(0.55));
var degraded = tree.and(ndvi2023.gte(0.25).and(ndvi2023.lt(0.55)))
                   .or(worldcover.eq(20)).or(worldcover.eq(30));

var rawMap = ee.Image(0)
  .where(dense,                                   1)
  .where(degraded,                                2)
  .where(worldcover.eq(80),                       3)
  .where(worldcover.eq(40).or(worldcover.eq(50)), 4)
  .where(worldcover.eq(60),                       5);

var classMap = rawMap.updateMask(rawMap.neq(0)).rename('landcover');

// ---------------------------------------------------------------------------
// Stratified random sampling (150 pts/class, seed=42)
// ---------------------------------------------------------------------------
var samples = classMap.stratifiedSample({
  numPoints: 150, classBand: 'landcover', region: studyArea,
  scale: 30, seed: 42, geometries: true, tileScale: 4
});


// ---------------------------------------------------------------------------
// Spatial block assignment
//
// Each sample point is assigned a 0.1-degree block id from its (bx, by)
// grid cell. Every DISTINCT block — not each point individually — is then
// assigned to train or holdout via a single seeded random draw per block.
// This guarantees every point inside a given 0.1-degree cell goes to the
// same partition, giving a genuine two-dimensional spatial split (rather
// than a random per-point split, which would let near-identical neighboring
// pixels leak between train and test).
// ---------------------------------------------------------------------------
samples = samples.map(function(f) {
  var c   = f.geometry().coordinates();
  var bx  = ee.Number(c.get(0)).multiply(10).floor();
  var by  = ee.Number(c.get(1)).multiply(10).floor();
  var bid = bx.format('%d').cat('_').cat(by.format('%d'));
  return f.set('block_id', bid);
});

// One random draw per DISTINCT block (seed=42), ~30% of blocks -> holdout.
var uniqueBlocks = samples.distinct('block_id').randomColumn('rand', 42);
var blockFolds = uniqueBlocks.map(function(b) {
  var isHoldout = ee.Number(b.get('rand')).gte(0.7);
  return b.set('block_fold', ee.Algorithms.If(isHoldout, 1, 0));
});

print('Number of distinct 0.1-deg blocks:', uniqueBlocks.size());
print('Blocks assigned to holdout (fold=1):',
  blockFolds.filter(ee.Filter.eq('block_fold', 1)).size());
print('Blocks assigned to training (fold=0):',
  blockFolds.filter(ee.Filter.eq('block_fold', 0)).size());

// Join each point back to its block's fold assignment.
var joinFilter = ee.Filter.equals({leftField: 'block_id', rightField: 'block_id'});
var joined = ee.Join.saveFirst('blockMatch').apply(samples, blockFolds, joinFilter);
samples = joined.map(function(f) {
  var match = ee.Feature(f.get('blockMatch'));
  return f.set('block_fold', match.get('block_fold'));
});


// ---------------------------------------------------------------------------
// Repeated K-fold block assignment (one random draw per DISTINCT block per
// seed; fold = floor(rand * K)). Same distinct-block logic as the holdout.
// ---------------------------------------------------------------------------
var K = 5;
var CV_SEEDS = [42, 7, 123, 2024, 99];
var foldProps = [];

// Lookup-dictionary assignment (block_id -> fold) instead of repeated joins:
// cheaper for Earth Engine, identical result.
var distinctBlocks = samples.distinct('block_id');
var foldDicts = {};
CV_SEEDS.forEach(function(seed) {
  var prop = 'fold_' + seed;
  foldProps.push(prop);
  var blocksK = distinctBlocks.randomColumn('r_cv', seed);
  foldDicts[prop] = ee.Dictionary.fromLists(
    blocksK.aggregate_array('block_id'),
    blocksK.aggregate_array('r_cv').map(function(r) {
      return ee.Number(r).multiply(K).floor().int();
    }));
});
samples = samples.map(function(f) {
  var bid = f.get('block_id');
  var out = f;
  CV_SEEDS.forEach(function(seed) {
    var prop = 'fold_' + seed;
    out = out.set(prop, foldDicts[prop].get(bid));
  });
  return out;
});

var sampled = composite2023.select(featureBands).sampleRegions({
  collection: samples,
  properties: ['landcover', 'block_id', 'block_fold'].concat(foldProps),
  scale: 30, tileScale: 8, geometries: false
}).filter(ee.Filter.notNull(featureBands));


// ---------------------------------------------------------------------------
// One train/test evaluation: z-scoring fitted on the training subset only
// (SVM), hyperparameters identical to Script 04. Returns the test points
// with p_rf / p_svm / p_cart predictions attached.
// ---------------------------------------------------------------------------
var zBands = featureBands.map(function(b) { return b + '_z'; });

function evaluateSplit(trainFC, testFC, rep, fold) {
  var means = featureBands.map(function(b) {
    return ee.Number(trainFC.aggregate_mean(b));
  });
  var sds = featureBands.map(function(b) {
    return ee.Number(trainFC.aggregate_total_sd(b)).max(0.000001);
  });
  function standardize(fc) {
    return fc.map(function(f) {
      var vals = featureBands.map(function(b, i) {
        return ee.Number(f.get(b)).subtract(means[i]).divide(sds[i]);
      });
      return f.set(ee.Dictionary.fromLists(zBands, vals));
    });
  }
  var trainZ = standardize(trainFC);
  var testZ  = standardize(testFC);

  var rf = ee.Classifier.smileRandomForest({numberOfTrees: 500, seed: 42})
    .train({features: trainFC, classProperty: 'landcover',
            inputProperties: featureBands});
  var cart = ee.Classifier.smileCart(50, 3)
    .train({features: trainFC, classProperty: 'landcover',
            inputProperties: featureBands});
  var svm = ee.Classifier.libsvm({kernelType: 'RBF', gamma: 0.05, cost: 10})
    .train({features: trainZ, classProperty: 'landcover',
            inputProperties: zBands});

  return testZ
    .classify(rf,   'p_rf')
    .classify(svm,  'p_svm')
    .classify(cart, 'p_cart')
    .map(function(f) {
      return ee.Feature(null, {
        rep: rep, fold: fold,
        block_id:  f.get('block_id'),
        landcover: f.get('landcover'),
        p_rf:  f.get('p_rf'),
        p_svm: f.get('p_svm'),
        p_cart: f.get('p_cart')
      });
    });
}

// (a) Original single holdout (rep = 0, fold = 0) — sanity check vs Script 04
var results = [evaluateSplit(
  sampled.filter(ee.Filter.eq('block_fold', 0)),
  sampled.filter(ee.Filter.eq('block_fold', 1)), 0, 0)];

// (b) Repeated 5-fold spatial-block CV
CV_SEEDS.forEach(function(seed) {
  var prop = 'fold_' + seed;
  for (var k = 0; k < K; k++) {
    results.push(evaluateSplit(
      sampled.filter(ee.Filter.neq(prop, k)),
      sampled.filter(ee.Filter.eq(prop, k)), seed, k));
  }
});

var allPreds = ee.FeatureCollection(results).flatten();

// NOTE: no console accuracy prints here (they time out interactively).
// The original-holdout sanity check (77.10 / 78.04 / 72.90 %) is printed
// by Python Script 13 from the exported CSV.

// Full results go to Drive (console printing of 3,964 rows would time out).
Export.table.toDrive({
  collection: allPreds,
  description: 'Script12_SpatialCV_Predictions',
  fileFormat: 'CSV',
  folder: 'Rangamati_Deforestation',
  selectors: ['rep', 'fold', 'block_id', 'landcover', 'p_rf', 'p_svm', 'p_cart']
});
