"""Sentinel Hub JS evalscripts for Sentinel-1 SAR, Sentinel-2 optical, and Copernicus DEM."""

DEM_ELEVATION = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["DEM"] }],
        output: { bands: 1, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    return [sample.DEM];
}
"""

S1_VV_VH_MEDIAN_DB = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["VV", "VH", "dataMask"] }],
        output: { bands: 3, sampleType: "FLOAT32" },
        mosaicking: "ORBIT"
    };
}
function evaluatePixel(samples) {
    var vv_vals = [], vh_vals = [];
    for (var s of samples) {
        if (s.dataMask && s.VV > 0) vv_vals.push(10 * Math.log10(s.VV));
        if (s.dataMask && s.VH > 0) vh_vals.push(10 * Math.log10(s.VH));
    }
    if (vv_vals.length === 0) return [-9999, -9999, 0];
    vv_vals.sort((a, b) => a - b);
    vh_vals.sort((a, b) => a - b);
    var mid = Math.floor(vv_vals.length / 2);
    var vv_med = vv_vals.length % 2 ? vv_vals[mid] : (vv_vals[mid-1] + vv_vals[mid]) / 2;
    var vh_med = vh_vals.length % 2 ? vh_vals[mid] : (vh_vals[mid-1] + vh_vals[mid]) / 2;
    return [vv_med, vh_med, 1];
}
"""

S1_VV_VH_SINGLE_DB = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["VV", "VH", "dataMask"] }],
        output: { bands: 3, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(s) {
    var vv_db = s.VV > 0 ? 10 * Math.log10(s.VV) : -9999;
    var vh_db = s.VH > 0 ? 10 * Math.log10(s.VH) : -9999;
    return [vv_db, vh_db, s.dataMask];
}
"""

# Sentinel-2 L2A bare-soil temporal median composite.
# Outputs 8 bands (float32, REFLECTANCE scale):
#   1=B02  2=B03  3=B04  4=B08  5=B8A  6=B11  7=B12  8=valid_mask
# Observations filtered: clouds (SCL 8/9/10), cloud shadows (3),
#   water (6), snow (11), saturated (1), no-data (0), and
#   dense vegetation (NDVI > 0.4) — bare/sparsely vegetated pixels only.
S2_BARE_SOIL_COMPOSITE = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02","B03","B04","B08","B8A","B11","B12","SCL","dataMask"],
            units: ["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE",
                    "REFLECTANCE","REFLECTANCE","REFLECTANCE","DN","DN"]
        }],
        output: { bands: 8, sampleType: "FLOAT32" },
        mosaicking: "ORBIT"
    };
}

var EXCLUDE_SCL = [0, 1, 2, 3, 6, 8, 9, 10, 11];

function evaluatePixel(samples) {
    var b02=[], b03=[], b04=[], b08=[], b8a=[], b11=[], b12=[];
    for (var s of samples) {
        if (!s.dataMask) continue;
        if (EXCLUDE_SCL.indexOf(s.SCL) >= 0) continue;
        var ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-9);
        if (ndvi > 0.4) continue;
        b02.push(s.B02); b03.push(s.B03); b04.push(s.B04);
        b08.push(s.B08); b8a.push(s.B8A); b11.push(s.B11); b12.push(s.B12);
    }
    if (b02.length === 0) return [-9999,-9999,-9999,-9999,-9999,-9999,-9999,0];
    function med(arr) {
        arr.sort(function(a,b){return a-b;});
        var m = Math.floor(arr.length / 2);
        return arr.length % 2 ? arr[m] : (arr[m-1] + arr[m]) / 2;
    }
    return [med(b02),med(b03),med(b04),med(b08),med(b8a),med(b11),med(b12),1];
}
"""
