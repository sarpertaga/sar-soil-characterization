"""Sentinel Hub JS evalscripts for Sentinel-1 SAR and Copernicus DEM."""

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
