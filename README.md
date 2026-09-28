# Copenhagen bicycle crash explorer

Interactive Streamlit/Folium app for exploring traffic conditions and predicted bicycle crash risk on 42,200 road segments. Clicking a road opens a local XGBoost SHAP explanation alongside the map.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## CARTO basemap key

The app reads `CARTO_API_KEY` from Streamlit secrets. For local development,
create `.streamlit/secrets.toml`:

```toml
CARTO_API_KEY = "paste-your-key-here"
```

On Streamlit Community Cloud, open the app settings, choose **Secrets**, and
add the same TOML entry there. Do not put the key in `app.py` or commit the
`secrets.toml` file. Without a key, the app falls back to OpenStreetMap tiles.

The app expects these existing files:

- `data/road_segments_with_predictions.gpkg`
- `models/final_best_model_B.pkl`
- `models/final_best_model_V.pkl`

The saved boosters do not contain feature names. `MODEL_FEATURES` in `app.py` records the recovered training order, validated by reproducing the probabilities stored in the GeoPackage.

## Data interpretation notes

- The supplied `car_oneway` field is identical to `car_traffic`. It is retained
  to preserve the trained feature order, but the app does not describe it as
  reliable one-way-road information.
- `cycling_allowed` is constant across the supplied road segments and is also
  retained only for model compatibility.
- The combined crash probability is calculated as
  `1 - (1 - p_bike) * (1 - p_vehicle)`. This assumes the two crash types are
  independent; it is not a third trained model.
- LTS values are read from the GeoPackage rather than recalculated. The current
  rule table has overlapping conditions and needs an explicit precedence rule
  before it can safely replace the stored classification.
