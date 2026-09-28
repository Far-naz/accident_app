from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import branca.colormap as cm
import folium
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "road_segments_slim.gpkg"
MODEL_PATHS = {
    "Bicycle–bicycle crash": ROOT / "models" / "final_best_model_B.pkl",
    "Bicycle–vehicle crash": ROOT / "models" / "final_best_model_V.pkl",
}

# The boosters were saved without feature names. This order was recovered from
# their split thresholds and validated against the probabilities in the GPKG.
MODEL_FEATURES = [
    "aadt_final_drive",
    "aadt_final_bike",
    "vehicle_bike_conflict_volume",
    "road_length_m",
    "road_rank",
    "maxspeed",
    "lanes",
    "car_oneway",
    "car_traffic",
    "lts",
    "bicycle_class",
    "cycling_allowed",
    "bikeinfra_both_sides",
    "traffic_signals_per_100m",
    "bus_route",
    "max_junction_degree",
    "num_adjacent_edges",
    "degree_asymmetry",
    "is_dead_end",
    "is_transition",
    "is_road_type_changed",
    "is_lts_changed",
    "street_bearing",
    "length_density",
    "is_bridge",
    "is_tunnel",
]

LAYER_OPTIONS = {
    "Level of traffic stress (LTS)": ("lts", "LTS level", "categorical_lts"),
    "Bicycle traffic flow": ("aadt_final_bike", "Bicycles/day", "YlGnBu_09"),
    "Vehicle traffic flow": ("aadt_final_drive", "Vehicles/day", "YlOrRd_09"),
    "Bike–vehicle conflict volume": (
        "vehicle_bike_conflict_volume",
        "Conflict volume",
        "PuRd_09",
    ),
    "Bicycle–bicycle crash probability": (
        "pred_prob_bike_accident",
        "Probability",
        "YlOrRd_09",
    ),
    "Bicycle–vehicle crash probability": (
        "pred_prob_vehicle_accident",
        "Probability",
        "YlOrRd_09",
    ),
    "Combined crash probability (assumes independence)": (
        "pred_prob_any_accident",
        "Probability",
        "YlOrRd_09",
    ),
}

LTS_COLORS = {
    1: "#22c55e",  # green
    2: "#f59e0b",  # amber
    3: "#dc2626",  # red
    4: "#991b1b",  # dark red
}

MUNICIPALITY_OPTIONS = {
    "All areas": None,
    "København": True,
    "Frederiksberg": False,
}

FRIENDLY_NAMES = {
    "aadt_final_drive": "Vehicle traffic",
    "aadt_final_bike": "Bicycle traffic",
    "vehicle_bike_conflict_volume": "Bike–vehicle exposure",
    "road_length_m": "Segment length",
    "road_rank": "Road rank",
    "maxspeed": "Speed limit",
    "lanes": "Number of lanes",
    # This legacy model input duplicates ``car_traffic`` in the supplied
    # GeoPackage. Keep the column name to preserve the trained feature order,
    # but do not present it to users as reliable one-way information.
    "car_oneway": "Motor-vehicle access (legacy field)",
    "car_traffic": "Cars permitted",
    "lts": "Traffic stress level",
    "bicycle_class": "Bicycle road class",
    "cycling_allowed": "Cycling permitted",
    "bikeinfra_both_sides": "Bike infrastructure both sides",
    "traffic_signals_per_100m": "Traffic signals / 100 m",
    "bus_route": "Bus route",
    "max_junction_degree": "Maximum junction degree",
    "num_adjacent_edges": "Adjacent road segments",
    "degree_asymmetry": "Junction asymmetry",
    "is_dead_end": "Dead end",
    "is_transition": "Network transition",
    "is_road_type_changed": "Road type change",
    "is_lts_changed": "Stress-level change",
    "street_bearing": "Street bearing",
    "length_density": "Road length density",
    "is_bridge": "Bridge",
    "is_tunnel": "Tunnel",
}


ROAD_RANKS = {
    1: "motorway",
    2: "trunk",
    3: "primary",
    4: "secondary",
    5: "tertiary",
    6: "unclassified",
    7: "residential",
    8: "living_street",
    9: "service",
    10: "pedestrian",
    11: "cycleway",
    12: "path",
    13: "footway",
    14: "steps"
}

BICYCLE_CLASSES = {
    1: "Protected bicycle track",
    2: "Bicycle lane",
    3: "Mixed road",
}

ROAD_NAME_CANDIDATES = [
    "Name",
]


class CalibratedModel:
    """Compatibility class for the calibrated models persisted by joblib.

    The attributes are populated by unpickling objects that were originally
    saved as ``__main__.CalibratedModel``.
    """

    model: xgb.Booster
    calibrator: Any
    model_type: str

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [feature for feature in MODEL_FEATURES if feature not in frame]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")
        matrix = xgb.DMatrix(frame[MODEL_FEATURES].astype(float).to_numpy())
        raw_probability = self.model.predict(matrix)
        calibrated = np.asarray(self.calibrator.predict(raw_probability), dtype=float)
        return np.clip(calibrated, 0.0, 1.0)


def validate_model(name: str, model: CalibratedModel) -> None:
    """Fail early when a persisted model is incompatible with this app."""

    missing_attributes = [
        attribute
        for attribute in ("model", "calibrator")
        if not hasattr(model, attribute)
    ]
    if missing_attributes:
        raise TypeError(
            f"{name} is missing persisted attributes: {', '.join(missing_attributes)}"
        )
    if not isinstance(model.model, xgb.Booster):
        raise TypeError(f"{name} does not contain an XGBoost Booster")
    if model.model.num_features() != len(MODEL_FEATURES):
        raise ValueError(
            f"{name} expects {model.model.num_features()} features; "
            f"the app defines {len(MODEL_FEATURES)}"
        )
    if not callable(getattr(model.calibrator, "predict", None)):
        raise TypeError(f"{name} does not contain a compatible probability calibrator")


def add_road_labels(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create a normalized display label from any available road-name field."""

    #source_column = next(
    #    (column for column in ROAD_NAME_CANDIDATES if column in roads.columns),
    #    None,
    #)
    #if source_column is None:
    #    roads["road_name"] = pd.Series(pd.NA, index=roads.index, dtype="string")
    #else:
    #    road_names = roads[source_column].astype("string").str.strip()
    #    roads["road_name"] = road_names.mask(
    #        road_names.isna()
    #        | road_names.eq("")
    #        | road_names.str.lower().isin({"nan", "none", "null", "<na>"})
    #    )
    roads["road_label"] = roads["Name"].fillna("Unnamed road")
    return roads


@st.cache_data(show_spinner="Loading road network…")
def load_roads() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads_projected = gpd.read_file(DATA_PATH).reset_index(drop=True)
    required_columns = set(MODEL_FEATURES) | {
        "geometry",
        "municipality_København",
        "vehicle_accidents",
        "bicycle_accidents",
        "pred_prob_bike_accident",
        "pred_prob_vehicle_accident",
        "pred_prob_any_accident",
    }
    missing_columns = sorted(required_columns.difference(roads_projected.columns))
    if missing_columns:
        raise ValueError(f"The road dataset is missing: {', '.join(missing_columns)}")
    roads_projected = add_road_labels(roads_projected)
    roads_projected["segment_id"] = roads_projected.index.astype(int)
    roads_map = roads_projected.to_crs(4326)
    # Roughly two metres in Copenhagen; reduces browser payload without visibly
    # changing road alignment at normal map zoom levels.
    roads_map.geometry = roads_map.geometry.simplify(0.00002, preserve_topology=True)
    return roads_projected, roads_map


@st.cache_resource(show_spinner="Loading crash models…")
def load_models() -> dict[str, CalibratedModel]:
    # The training notebook serialized this class as ``__main__.CalibratedModel``.
    # Register the compatibility class there so loading also works when this file
    # is imported (tests, multipage apps, or alternative Streamlit launchers).
    main_module = sys.modules["__main__"]
    previous_class = getattr(main_module, "CalibratedModel", None)
    main_module.CalibratedModel = CalibratedModel
    try:
        models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
        for name, model in models.items():
            validate_model(name, model)
        return models
    finally:
        if previous_class is None:
            delattr(main_module, "CalibratedModel")
        else:
            main_module.CalibratedModel = previous_class


def color_scale(values: pd.Series, palette_name: str, caption: str):
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError(f"The selected layer has no finite values: {caption}")
    low = float(finite.min())
    high = float(finite.quantile(0.99))
    if high <= low:
        high = float(finite.max()) or low + 1.0
    palette = getattr(cm.linear, palette_name)
    scale = palette.scale(low, high)
    scale.caption = caption
    return scale, low, high


def make_map(
    roads_map: gpd.GeoDataFrame,
    layer_label: str,
    selected_id: int | None,
) -> folium.Map:
    field, unit, palette = LAYER_OPTIONS[layer_label]
    is_lts_layer = field == "lts"
    if is_lts_layer:
        scale, low, high = None, 1.0, 4.0
    else:
        scale, low, high = color_scale(roads_map[field], palette, layer_label)

    bounds = roads_map.total_bounds
    selected = roads_map.loc[roads_map["segment_id"] == selected_id]
    if selected.empty:
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        initial_zoom = 12
    else:
        selected_bounds = selected.total_bounds
        center = [
            (selected_bounds[1] + selected_bounds[3]) / 2,
            (selected_bounds[0] + selected_bounds[2]) / 2,
        ]
        initial_zoom = 16
    road_map = folium.Map(
        location=center,
        zoom_start=initial_zoom,
        tiles="CartoDB positron",
        prefer_canvas=True,
        control_scale=True,
    )

    display = roads_map[["segment_id", "road_label", field, "geometry"]].copy()
    display["map_value"] = pd.to_numeric(display[field], errors="coerce").fillna(low)
    display["map_value"] = display["map_value"].clip(low, high)
    if is_lts_layer:
        display["display_value"] = display[field].map(lambda value: f"LTS {int(value)}")
    else:
        display["display_value"] = display[field].map(
            lambda value: (
                f"{value:.2%}" if "probability" in layer_label.lower() else f"{value:,.1f}"
            )
        )

    def style(feature):
        props = feature["properties"]
        segment_id = int(props["segment_id"])
        if selected_id == segment_id:
            return {"color": "#facc15", "weight": 10, "opacity": 1}
        if is_lts_layer:
            return {
                "color": LTS_COLORS.get(int(props["map_value"]), "#94a3b8"),
                "weight": 4,
                "opacity": 0.9,
            }
        return {
            "color": scale(float(props["map_value"])),
            "weight": 3,
            "opacity": 0.82,
        }

    folium.GeoJson(
        display.to_json(drop_id=True),
        name=layer_label,
        style_function=style,
        highlight_function=lambda _: {"weight": 6, "opacity": 1},
        tooltip=folium.GeoJsonTooltip(
            fields=[ "display_value"],
            aliases=[ unit],
            sticky=False,
        ),
        popup=folium.GeoJsonPopup(
            fields=[ "display_value"],
            aliases=[ unit],
        ),
        zoom_on_click=False,
    ).add_to(road_map)
    if is_lts_layer:
        legend_items = "".join(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="width:22px;height:5px;background:{color};display:inline-block;"></span>'
            f'<span>LTS {level}</span></div>'
            for level, color in LTS_COLORS.items()
        )
        legend = f"""
        <div style="position:fixed;bottom:28px;right:18px;z-index:9999;
                    background:white;padding:10px 14px;border:1px solid #cbd5e1;
                    border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.18);
                    font-size:13px;color:#0f172a;">
          <div style="font-weight:700;margin-bottom:5px;">Traffic stress</div>
          {legend_items}
        </div>
        """
        road_map.get_root().html.add_child(folium.Element(legend))
    else:
        scale.add_to(road_map)

    if selected.empty:
        road_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    else:
        selected_geometry = selected.geometry.iloc[0]
        midpoint = selected_geometry.interpolate(0.5, normalized=True)
        folium.CircleMarker(
            location=[midpoint.y, midpoint.x],
            radius=7,
            color="#111827",
            weight=3,
            fill=True,
            fill_color="#facc15",
            fill_opacity=1,
            tooltip=f"Selected road: {selected['road_label'].iloc[0]}",
        ).add_to(road_map)
    return road_map


def clicked_coordinates(map_state: dict | None) -> tuple[float, float] | None:
    if not map_state:
        return None
    for key in ("last_object_clicked", "last_clicked"):
        point = map_state.get(key)
        if point and point.get("lat") is not None and point.get("lng") is not None:
            return float(point["lat"]), float(point["lng"])
    return None


def nearest_segment(
    roads_projected: gpd.GeoDataFrame,
    latitude: float,
    longitude: float,
    maximum_distance_m: float = 45,
) -> int | None:
    point = gpd.GeoSeries(gpd.points_from_xy([longitude], [latitude]), crs=4326).to_crs(
        roads_projected.crs
    ).iloc[0]
    spatial_index = roads_projected.sindex
    candidates = list(spatial_index.query(point.buffer(maximum_distance_m), predicate="intersects"))
    if not candidates:
        return None
    distances = roads_projected.geometry.iloc[candidates].distance(point)
    closest_position = int(np.argmin(distances.to_numpy()))
    if float(distances.iloc[closest_position]) > maximum_distance_m:
        return None
    return int(roads_projected.iloc[candidates[closest_position]]["segment_id"])


def shap_contributions(model: CalibratedModel, row: pd.Series) -> tuple[np.ndarray, float]:
    matrix = xgb.DMatrix(row[MODEL_FEATURES].astype(float).to_numpy().reshape(1, -1))
    contributions = model.model.predict(matrix, pred_contribs=True)[0]
    return contributions[:-1], float(contributions[-1])


@st.cache_data(show_spinner="Calculating SHAP values for the road network\u2026")
def absolute_shap_values(model_name: str) -> np.ndarray:
    """Return per-road absolute SHAP values in segment-id order."""

    roads_projected, _ = load_roads()
    model = load_models()[model_name]
    matrix = xgb.DMatrix(roads_projected[MODEL_FEATURES].astype(float).to_numpy())
    contributions = model.model.predict(matrix, pred_contribs=True)
    return np.abs(contributions[:, :-1])


def average_shap_chart(
    roads: gpd.GeoDataFrame,
    model_name: str,
    top_n: int = 10,
) -> go.Figure:
    """Plot mean absolute SHAP values for the currently filtered roads."""

    segment_ids = roads["segment_id"].astype(int).to_numpy()
    means = absolute_shap_values(model_name)[segment_ids].mean(axis=0)
    order = np.argsort(means)[::-1][:top_n]
    chart = pd.DataFrame(
        {
            "Feature": [FRIENDLY_NAMES[MODEL_FEATURES[index]] for index in order],
            "Mean absolute SHAP": means[order],
        }
    ).sort_values("Mean absolute SHAP")
    fig = go.Figure(
        go.Bar(
            x=chart["Mean absolute SHAP"],
            y=chart["Feature"],
            orientation="h",
            marker_color="#6366f1",
            hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=390,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title="Mean absolute SHAP value (log-odds)",
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def contribution_chart(values: np.ndarray, row: pd.Series, top_n: int = 10) -> go.Figure:
    order = np.argsort(np.abs(values))[::-1][:top_n]
    chart = pd.DataFrame(
        {
            "Feature": [FRIENDLY_NAMES[MODEL_FEATURES[index]] for index in order],
            "Contribution": values[order],
            "Value": [row[MODEL_FEATURES[index]] for index in order],
        }
    ).sort_values("Contribution")
    colors = np.where(chart["Contribution"] >= 0, "#e45756", "#2a9d8f")
    fig = go.Figure(
        go.Bar(
            x=chart["Contribution"],
            y=chart["Feature"],
            orientation="h",
            marker_color=colors,
            customdata=chart[["Value"]],
            hovertemplate="%{y}<br>SHAP: %{x:.3f}<br>Road value: %{customdata[0]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#64748b")
    fig.update_layout(
        height=390,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title="Contribution to uncalibrated model score (log-odds)",
        yaxis_title=None,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def contribution_summary(values: np.ndarray) -> str:
    positive = np.flatnonzero(values > 0)
    negative = np.flatnonzero(values < 0)

    if positive.size:
        strongest_positive = positive[np.argmax(values[positive])]
        increase_text = FRIENDLY_NAMES[MODEL_FEATURES[strongest_positive]].lower()
    else:
        increase_text = "no individual feature"

    if negative.size:
        strongest_negative = negative[np.argmin(values[negative])]
        decrease_text = FRIENDLY_NAMES[MODEL_FEATURES[strongest_negative]].lower()
    else:
        decrease_text = "no individual feature"

    return (
        f"For this segment, **{increase_text}** has the strongest upward effect on the "
        f"underlying model score, while **{decrease_text}** has the strongest downward effect."
    )


def format_value(field: str, value) -> str:
    if pd.isna(value):
        return "Not available"
    if field.startswith("pred_prob"):
        return f"{float(value):.2%}"
    if field in {"aadt_final_drive", "aadt_final_bike"}:
        return f"{float(value):,.0f}/day"
    if field == "lts":
        return f"LTS {int(value)}"
    return f"{float(value):,.2f}"


def yes_no(value) -> str:
    if pd.isna(value):
        return "Not available"
    return "Yes" if bool(value) else "No"


def road_rank_name(value) -> str:
    if pd.isna(value):
        return "Not available"
    name = ROAD_RANKS.get(int(value))
    return name.replace("_", " ").title() if name else "Unknown"


def render_explanation(row: pd.Series, models: dict[str, CalibratedModel]) -> None:
    st.markdown(f"### Segment {int(row['segment_id']):,}")
    model_name = st.radio(
        "Explain model", list(models), horizontal=True, key="segment_explanation_model"
    )
    probability_field = (
        "pred_prob_bike_accident"
        if model_name == "Bicycle–bicycle crash"
        else "pred_prob_vehicle_accident"
    )
    probability = float(row[probability_field])
    st.metric("Predicted crash probability", f"{probability:.2%}")

    contributions, base_value = shap_contributions(models[model_name], row)
    st.info(
        "The percentage above is the calibrated probability. The chart explains the "
        "underlying XGBoost score before calibration: red features raise the score, "
        "teal features lower it, and longer bars indicate a stronger influence."
    )
    st.markdown(contribution_summary(contributions))
    st.plotly_chart(contribution_chart(contributions, row), width="stretch")
    st.caption(
        f"Uncalibrated baseline score: {base_value:.3f} log-odds. "
        "SHAP contributions are not additive on the calibrated probability scale."
    )

    with st.expander("Road attributes", expanded=False):
        attributes = {
            "Bicycle traffic": format_value("aadt_final_bike", row["aadt_final_bike"]),
            "Vehicle traffic": format_value("aadt_final_drive", row["aadt_final_drive"]),
            "Bicycle–vehicle exposure index": format_value(
                "vehicle_bike_conflict_volume", row["vehicle_bike_conflict_volume"]
            ),
            "Historical bicycle–vehicle crashes": str(int(row["vehicle_accidents"])),
            "Historical bicycle–bicycle crashes": str(int(row["bicycle_accidents"])),
            "Historical crashes (total)": str(
                int(row["vehicle_accidents"] + row["bicycle_accidents"])
            ),
            "Speed limit": f"{row['maxspeed']:.0f} km/h",
            "Number of lanes": f"{row['lanes']:.0f}",
            "Length": f"{row['road_length_m']:.0f} m",
            "Traffic stress": format_value("lts", row["lts"]),
            "Bicycle–bicycle risk": format_value(
                "pred_prob_bike_accident", row["pred_prob_bike_accident"]
            ),
            "Bicycle–vehicle risk": format_value(
                "pred_prob_vehicle_accident", row["pred_prob_vehicle_accident"]
            ),
            "Road class": road_rank_name(row["road_rank"]),
            "Bicycle infrastructure": BICYCLE_CLASSES.get(
                int(row["bicycle_class"]), "Unknown"
            ),
            "Motor vehicles permitted": yes_no(row["car_traffic"]),
            "Bike infrastructure on both sides": yes_no(row["bikeinfra_both_sides"]),
            "Traffic signals / 100 m": f"{row['traffic_signals_per_100m']:.2f}",
            "Bus route": yes_no(row["bus_route"]),
            "Dead end": yes_no(row["is_dead_end"]),
            "Network transition": yes_no(row["is_transition"]),
            "Road type change": yes_no(row["is_road_type_changed"]),
            "Stress-level change": yes_no(row["is_lts_changed"]),
            "Bridge": yes_no(row["is_bridge"]),
            "Tunnel": yes_no(row["is_tunnel"]),

        }
        st.dataframe(
            pd.DataFrame(attributes.items(), columns=["Attribute", "Value"]),
            hide_index=True,
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title="Copenhagen Bicycle Crash Explorer", page_icon="🚲", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 1rem;}
        [data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0;
                                  padding:12px 16px; border-radius:12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Copenhagen bicycle crash explorer")
    st.caption("Explore road conditions and modelled crash risk. Click a road to understand its prediction.")

    if not DATA_PATH.exists() or not all(path.exists() for path in MODEL_PATHS.values()):
        st.error("The GeoPackage or one of the model files is missing. Check the data/ and models/ folders.")
        st.stop()

    roads_projected, roads_map = load_roads()
    models = load_models()
    layer_label = st.selectbox("Map layer", list(LAYER_OPTIONS), index=4)

    available_lts = sorted(roads_projected["lts"].astype(int).unique().tolist())
    available_road_ranks = sorted(
        roads_projected["road_rank"].astype(int).unique().tolist()
    )
    road_rank_labels = {rank: road_rank_name(rank) for rank in available_road_ranks}
    available_speeds = sorted(
        roads_projected["maxspeed"].astype(int).unique().tolist()
    )

    with st.expander("Filter roads", expanded=True), st.form("road_filters"):
        municipality_control, lts_control, road_type_control, speed_control = st.columns(4)
        with municipality_control:
            municipality_label = st.selectbox(
                "Municipality", list(MUNICIPALITY_OPTIONS)
            )
        with lts_control:
            selected_lts = st.multiselect(
                "LTS levels",
                available_lts,
                default=available_lts,
                format_func=lambda value: f"LTS {value}",
            )
        with road_type_control:
            selected_road_ranks = st.multiselect(
                "Road types",
                available_road_ranks,
                default=available_road_ranks,
                format_func=lambda value: road_rank_labels[value],
            )
        with speed_control:
            selected_speeds = st.multiselect(
                "Speed limits",
                available_speeds,
                default=available_speeds,
                format_func=lambda value: f"{value} km/h",
            )

        vehicle_probability_control, bicycle_probability_control = st.columns(2)
        with vehicle_probability_control:
            vehicle_probability_range = st.slider(
                "Bicycle–vehicle crash probability",
                min_value=0.0,
                max_value=100.0,
                value=(0.0, 100.0),
                step=0.1,
                format="%.1f%%",
            )
        with bicycle_probability_control:
            bicycle_probability_range = st.slider(
                "Bicycle–bicycle crash probability",
                min_value=0.0,
                max_value=100.0,
                value=(0.0, 100.0),
                step=0.1,
                format="%.1f%%",
            )
        st.form_submit_button("Apply filters", type="primary")

    municipality_value = MUNICIPALITY_OPTIONS[municipality_label]
    if municipality_value is None:
        filtered_projected = roads_projected
        filtered_map = roads_map
    else:
        municipality_mask = roads_projected["municipality_København"].eq(municipality_value)
        filtered_projected = roads_projected.loc[municipality_mask]
        filtered_map = roads_map.loc[municipality_mask]

    filter_mask = (
        filtered_projected["lts"].astype(int).isin(selected_lts)
        & filtered_projected["road_rank"].astype(int).isin(selected_road_ranks)
        & filtered_projected["maxspeed"].astype(int).isin(selected_speeds)
        & filtered_projected["pred_prob_vehicle_accident"].mul(100).between(
            *vehicle_probability_range
        )
        & filtered_projected["pred_prob_bike_accident"].mul(100).between(
            *bicycle_probability_range
        )
    )
    filtered_projected = filtered_projected.loc[filter_mask]
    filtered_map = filtered_map.loc[filter_mask]

    selected_id = st.session_state.get("selected_segment")
    if selected_id is not None and selected_id not in set(filtered_projected["segment_id"]):
        st.session_state.selected_segment = None
        selected_id = None

    map_column, explanation_column = st.columns([1.9, 1], gap="large")
    with map_column:
        st.caption(f"Showing {len(filtered_map):,} road segments in {municipality_label}.")
        if filtered_map.empty:
            st.warning("No road segments match the selected filters.")
        else:
            road_map = make_map(filtered_map, layer_label, selected_id)
            map_state = st_folium(
                road_map,
                height=690,
                use_container_width=True,
                returned_objects=["last_object_clicked", "last_clicked"],
                key=(
                    f"road_map_{layer_label}_{municipality_label}_"
                    f"{tuple(selected_lts)}_{tuple(selected_road_ranks)}_"
                    f"{tuple(selected_speeds)}_{vehicle_probability_range}_"
                    f"{bicycle_probability_range}"
                ),
            )
            coordinates = clicked_coordinates(map_state)
            if coordinates:
                click_signature = (round(coordinates[0], 7), round(coordinates[1], 7))
                if click_signature != st.session_state.get("last_click"):
                    st.session_state.last_click = click_signature
                    match = nearest_segment(filtered_projected, *coordinates)
                    if match is not None:
                        st.session_state.selected_segment = match
                        st.rerun()
                    else:
                        st.toast("Click closer to a road segment.")

    with explanation_column:
        st.markdown("### Average feature importance")
        if filtered_projected.empty:
            st.info("Apply broader filters to calculate average absolute SHAP values.")
        else:
            average_model = st.radio(
                "SHAP model",
                list(models),
                horizontal=True,
                key="average_shap_model",
            )
            st.caption(
                f"Mean absolute SHAP values across {len(filtered_projected):,} visible "
                "road segments. Larger values indicate greater average influence, "
                "regardless of direction."
            )
            st.plotly_chart(
                average_shap_chart(filtered_projected, average_model),
                width="stretch",
                key=f"average_shap_{average_model}_{len(filtered_projected)}",
            )

        selected_id = st.session_state.get("selected_segment")
        if selected_id is None:
            st.info("Click a coloured road segment for its individual SHAP explanation.")
        else:
            selected_row = roads_projected.loc[
                roads_projected["segment_id"] == selected_id
            ].iloc[0]
            st.divider()
            render_explanation(selected_row, models)


if __name__ == "__main__":
    main()
