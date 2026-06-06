# SAR-Based Soil Characterization

Geospatial intelligence framework for characterizing soil properties and surface conditions using Sentinel-1 SAR, terrain analysis, and machine learning.

## Versions

| Version | Theme | Status |
|---|---|---|
| V1 | Exploratory soil characterization (SAR + terrain + hydrology) | 🔄 In progress |
| V2 | Soil property modelling (ML regression) | Planned |
| V3 | GeoAI soil intelligence (deep learning) | Planned |
| V4 | Interactive soil intelligence platform | Planned |

## V1 Stack

- **SAR data:** Sentinel Hub Processing API (Sentinel-1 IW σ⁰ VV+VH)
- **Terrain:** Copernicus DEM GLO-30 via GDAL
- **Hydrology:** WhiteboxTools (TWI, SPI, flow accumulation)
- **Classification:** 7-class soil hardness (calibrated with 209 pile tests, Ağrı + Kars)
- **Pilot AOI:** Konya Closed Basin

## Quickstart

```bash
git clone https://github.com/sarpertaga/sar-soil-characterization
cd sar-soil-characterization
mamba env create -f environment.yml
conda activate soilgeo
pip install -e .
cp .env.example .env   # fill in SH_CLIENT_ID + SH_CLIENT_SECRET
python pipelines/run_v1.py --stages all
```

## License

MIT
