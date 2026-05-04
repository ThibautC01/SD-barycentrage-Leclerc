from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(
    page_title="Simulation entrepots Leclerc - vGroup",
    page_icon="🏬",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parent
EXCLUDED_SCAS = {"GIE MASCAREIGNE", "GIE GIELEC", "SODIPLEC"}
NORD_WAREHOUSE = {
    "name": "Entrepot Nord - Ressons-sur-Matz",
    "lat": 49.5380,
    "lon": 2.7751,
}
FIXED_SOUTH_OPTIONS = {
    "Toulouse": {"name": "Entrepot Sud - Toulouse", "lat": 43.6045, "lon": 1.4440},
    "Avignon": {"name": "Entrepot Sud - Avignon", "lat": 43.9493, "lon": 4.8055},
}
DELIVERY_CENTRAL = "Livraison centrale"
DELIVERY_STORE = "Livraison magasin"
DELIVERY_STORE_SCA_BARYCENTER = "Livraison magasin barycentree SCA"
DELIVERY_OPTIONS = (DELIVERY_CENTRAL, DELIVERY_STORE, DELIVERY_STORE_SCA_BARYCENTER)
WEIGHT_COL = "Poids - CA"
SCA_COL = "Libelle SCA"
STORE_ID_COL = "Identifiant fictif"
Warehouse = Dict[str, Union[float, str]]

# Coordonnees manuelles des centrales, basees sur l'onglet recap.
CENTRALS = {
    "LECASUD": {
        "label": "LECASUD - Le Luc",
        "address": "ZI des Lauves, 83340 Le Luc",
        "lat": 43.3948,
        "lon": 6.3124,
    },
    "SCACENTRE": {
        "label": "SCACENTRE - Yzeure",
        "address": "10 Rue Colbert, 03400 Yzeure",
        "lat": 46.5657,
        "lon": 3.3546,
    },
    "SCACHAP": {
        "label": "SCACHAP - Ruffec",
        "address": "ZI de la Gare, 16700 Ruffec",
        "lat": 46.0307,
        "lon": 0.2003,
    },
    "SCADIF": {
        "label": "SCADIF - Reau",
        "address": "Parc d'activite de l'A5, 77750 Reau",
        "lat": 48.5883,
        "lon": 2.6274,
    },
    "SCALANDES": {
        "label": "SCALANDES - Mont-de-Marsan",
        "address": "ZA de Pemegnan, 40001 Mont-de-Marsan",
        "lat": 43.8921,
        "lon": -0.5000,
    },
    "SCANORMANDE": {
        "label": "SCANORMANDE - Lisieux",
        "address": "ZI Nord, 14100 Lisieux",
        "lat": 49.1460,
        "lon": 0.2293,
    },
    "SCAOUEST": {
        "label": "SCAOUEST - Saint-Etienne-de-Montluc",
        "address": "Route de Cordemais, 44360 Saint-Etienne-de-Montluc",
        "lat": 47.2743,
        "lon": -1.7804,
    },
    "SCAPALSACE": {
        "label": "SCAPALSACE - Colmar",
        "address": "ZI Nord, 68025 Colmar",
        "lat": 48.0790,
        "lon": 7.3550,
    },
    "SCAPARTOIS": {
        "label": "SCAPARTOIS - Tilloy-les-Mofflaines",
        "address": "ZI Arras Est, 62217 Tilloy-les-Mofflaines",
        "lat": 50.2827,
        "lon": 2.8223,
    },
    "SCAPEST": {
        "label": "SCAPEST - Saint-Martin-sur-le-Pre",
        "address": "ZI du Moulin, 51039 Chalons-en-Champagne",
        "lat": 48.9724,
        "lon": 4.3630,
    },
    "SCAPNOR": {
        "label": "SCAPNOR - Bruyeres-sur-Oise",
        "address": "ZAE Chemin du bac des Aubins, 95820 Bruyeres-sur-Oise",
        "lat": 49.1540,
        "lon": 2.3264,
    },
    "SCARMOR": {
        "label": "SCARMOR - Landerneau",
        "address": "29419 Landerneau Cedex",
        "lat": 48.4516,
        "lon": -4.2517,
    },
    "SCASO": {
        "label": "SCASO - Cestas",
        "address": "ZI de Toctoucau, 33612 Cestas Cedex",
        "lat": 44.7452,
        "lon": -0.6813,
    },
    "SOCAMAINE": {
        "label": "SOCAMAINE - Champagne",
        "address": "Route de Paris, 72470 Champagne",
        "lat": 48.0224,
        "lon": 0.3320,
    },
    "SOCAMIL": {
        "label": "SOCAMIL - Castelnaudary",
        "address": "511 Av. Gerard Rouviere, 11400 Castelnaudary",
        "lat": 43.3189,
        "lon": 1.9539,
    },
    "SOCARA": {
        "label": "SOCARA - Villette-d'Anthon",
        "address": "6 rue du Marais, 38280 Villette-d'Anthon",
        "lat": 45.7941,
        "lon": 5.1170,
    },
}


@dataclass
class SimulationResult:
    points_df: pd.DataFrame
    export_df: pd.DataFrame
    warehouse_summary: pd.DataFrame
    distances_summary: pd.DataFrame
    south_warehouse: Warehouse
    total_weighted_distance: float


def find_excel_path(prefix: str) -> Path:
    matches = sorted(BASE_DIR.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"Impossible de trouver un fichier commencant par {prefix!r}.")
    return matches[0]


def find_first_excel_path(prefixes: Iterable[str]) -> Path:
    checked = []
    for prefix in prefixes:
        checked.append(prefix)
        matches = sorted(BASE_DIR.glob(f"{prefix}*.xlsx"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Impossible de trouver un fichier commencant par "
        + ", ".join(repr(prefix) for prefix in checked)
        + "."
    )


@st.cache_data(show_spinner=False)
def load_store_data() -> Tuple[pd.DataFrame, Path]:
    path = find_excel_path("Liste_magasins")
    df = pd.read_excel(path, sheet_name="Sheet1").copy()

    required_columns = {
        SCA_COL,
        STORE_ID_COL,
        "Latitude",
        "Longitude",
        WEIGHT_COL,
        "Raison sociale",
        "Ville",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {', '.join(missing_columns)}")

    df["row_id"] = np.arange(len(df))
    filtered = df.loc[~df[SCA_COL].isin(EXCLUDED_SCAS)].copy()
    filtered["Latitude"] = pd.to_numeric(filtered["Latitude"], errors="coerce")
    filtered["Longitude"] = pd.to_numeric(filtered["Longitude"], errors="coerce")
    filtered[WEIGHT_COL] = pd.to_numeric(filtered[WEIGHT_COL], errors="coerce")
    filtered = filtered.dropna(subset=["Latitude", "Longitude", WEIGHT_COL])
    filtered["Nom affichage"] = filtered["Raison sociale"].fillna(filtered["Ville"].astype(str))
    return filtered, path


@st.cache_data(show_spinner=False)
def load_recap_data() -> Tuple[pd.DataFrame, Path]:
    path = find_first_excel_path(("liste entrepôts", "liste entrepots"))
    df = pd.read_excel(path, sheet_name="recap")
    return df, path


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius = 6371.0088
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def weighted_centroid(df: pd.DataFrame) -> Tuple[float, float]:
    weights = df[WEIGHT_COL].to_numpy()
    lat = np.average(df["Latitude"].to_numpy(), weights=weights)
    lon = np.average(df["Longitude"].to_numpy(), weights=weights)
    return float(lat), float(lon)


def build_demand_points(stores_df: pd.DataFrame, delivery_modes: Dict[str, str]) -> pd.DataFrame:
    records: List[dict] = []
    for sca, group in stores_df.groupby(SCA_COL, sort=True):
        mode = delivery_modes.get(sca, DELIVERY_CENTRAL)
        if mode == DELIVERY_CENTRAL:
            central = CENTRALS[sca]
            records.append(
                {
                    "point_id": f"CENTRALE::{sca}",
                    "display_name": central["label"],
                    "display_type": "Centrale",
                    SCA_COL: sca,
                    "Latitude": central["lat"],
                    "Longitude": central["lon"],
                    WEIGHT_COL: group[WEIGHT_COL].sum(),
                    "store_count": int(len(group)),
                    "delivery_mode": mode,
                    "store_row_ids": tuple(group["row_id"].tolist()),
                    "store_ids": tuple(group[STORE_ID_COL].astype(str).tolist()),
                }
            )
        elif mode == DELIVERY_STORE_SCA_BARYCENTER:
            lat, lon = weighted_centroid(group)
            records.append(
                {
                    "point_id": f"BARY_SCA::{sca}",
                    "display_name": f"{sca} - Barycentre magasins",
                    "display_type": "Barycentre SCA",
                    SCA_COL: sca,
                    "Latitude": lat,
                    "Longitude": lon,
                    WEIGHT_COL: group[WEIGHT_COL].sum(),
                    "store_count": int(len(group)),
                    "delivery_mode": mode,
                    "store_row_ids": tuple(group["row_id"].tolist()),
                    "store_ids": tuple(group[STORE_ID_COL].astype(str).tolist()),
                }
            )
        else:
            for row in group.to_dict(orient="records"):
                records.append(
                    {
                        "point_id": f"MAG::{row['row_id']}",
                        "display_name": row["Nom affichage"],
                        "display_type": "Magasin",
                        SCA_COL: row[SCA_COL],
                        "Latitude": float(row["Latitude"]),
                        "Longitude": float(row["Longitude"]),
                        WEIGHT_COL: float(row[WEIGHT_COL]),
                        "store_count": 1,
                        "delivery_mode": mode,
                        "store_row_ids": (int(row["row_id"]),),
                        "store_ids": (str(row[STORE_ID_COL]),),
                    }
                )
    return pd.DataFrame(records).sort_values("point_id").reset_index(drop=True)


def compute_axis_projection(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    north_lat: float,
    north_lon: float,
    south_lat: float,
    south_lon: float,
) -> np.ndarray:
    ref_lat = np.radians((north_lat + south_lat) / 2.0)
    lon_scale = np.cos(ref_lat)
    north_x, north_y = north_lon * lon_scale, north_lat
    south_x, south_y = south_lon * lon_scale, south_lat
    axis = np.array([north_x - south_x, north_y - south_y], dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return latitudes.copy()
    axis_unit = axis / axis_norm
    x = longitudes * lon_scale
    y = latitudes
    vectors = np.column_stack([x - south_x, y - south_y])
    return vectors @ axis_unit


def solve_north_assignment(
    points_df: pd.DataFrame,
    north_cost: np.ndarray,
    south_cost: np.ndarray,
    north_target: int,
    south_warehouse: Warehouse,
) -> np.ndarray:
    total_stores = int(points_df["store_count"].sum())

    if north_target <= 0:
        return np.zeros(len(points_df), dtype=bool)
    if north_target >= total_stores:
        return np.ones(len(points_df), dtype=bool)

    projections = compute_axis_projection(
        points_df["Latitude"].to_numpy(),
        points_df["Longitude"].to_numpy(),
        float(NORD_WAREHOUSE["lat"]),
        float(NORD_WAREHOUSE["lon"]),
        float(south_warehouse["lat"]),
        float(south_warehouse["lon"]),
    )

    ordered = points_df.copy()
    ordered["projection"] = projections
    ordered["north_cost"] = north_cost
    ordered["south_cost"] = south_cost
    ordered = ordered.sort_values("projection", ascending=False).reset_index()

    cum_store_count = ordered["store_count"].cumsum().to_numpy()
    cum_north_cost = ordered["north_cost"].cumsum().to_numpy()
    south_suffix = ordered["south_cost"][::-1].cumsum()[::-1].to_numpy()
    total_south_cost = float(ordered["south_cost"].sum())

    candidate_prefixes = {0, len(ordered)}
    for idx, store_count in enumerate(cum_store_count):
        if int(store_count) <= north_target:
            candidate_prefixes.add(idx + 1)
        else:
            candidate_prefixes.add(idx + 1)
            candidate_prefixes.add(idx)
            break

    best_prefix = 0
    best_gap = total_stores
    best_total_cost = total_south_cost

    for prefix in sorted(p for p in candidate_prefixes if 0 <= p <= len(ordered)):
        assigned_store_count = 0 if prefix == 0 else int(cum_store_count[prefix - 1])
        gap = abs(assigned_store_count - north_target)
        candidate_total = 0.0 if prefix == 0 else float(cum_north_cost[prefix - 1])
        if prefix < len(ordered):
            candidate_total += float(south_suffix[prefix])
        if gap < best_gap or (gap == best_gap and candidate_total < best_total_cost - 1e-12):
            best_gap = gap
            best_total_cost = candidate_total
            best_prefix = prefix

    assignment = np.zeros(len(points_df), dtype=bool)
    if best_prefix > 0:
        assignment[ordered.loc[: best_prefix - 1, "index"].to_list()] = True
    return assignment


def assign_points(points_df: pd.DataFrame, south_warehouse: Warehouse, north_ratio_pct: float) -> pd.DataFrame:
    total_stores = int(points_df["store_count"].sum())
    north_target = int(round(total_stores * north_ratio_pct / 100.0))

    lat = points_df["Latitude"].to_numpy()
    lon = points_df["Longitude"].to_numpy()
    weights = points_df[WEIGHT_COL].to_numpy()

    north_dist = haversine_km(lat, lon, float(NORD_WAREHOUSE["lat"]), float(NORD_WAREHOUSE["lon"]))
    south_dist = haversine_km(lat, lon, float(south_warehouse["lat"]), float(south_warehouse["lon"]))

    north_cost = north_dist * weights
    south_cost = south_dist * weights
    north_mask = solve_north_assignment(points_df, north_cost, south_cost, north_target, south_warehouse)

    assigned = points_df.copy()
    assigned["assigned_warehouse"] = np.where(north_mask, "Nord", "Sud")
    assigned["distance_km"] = np.where(north_mask, north_dist, south_dist)
    assigned["weighted_distance"] = np.where(north_mask, north_cost, south_cost)
    assigned["north_distance_km"] = north_dist
    assigned["south_distance_km"] = south_dist
    assigned["north_cost"] = north_cost
    assigned["south_cost"] = south_cost
    return assigned


def compute_barycentric_solution(points_df: pd.DataFrame, north_ratio_pct: float) -> Tuple[pd.DataFrame, Warehouse]:
    lat, lon = weighted_centroid(points_df)
    south: Warehouse = {"name": "Entrepot Sud - Barycentre mathematique", "lat": lat, "lon": lon}
    previous_assignment = None

    for _ in range(30):
        assigned = assign_points(points_df, south, north_ratio_pct)
        current_assignment = assigned["assigned_warehouse"].tolist()
        south_points = assigned.loc[assigned["assigned_warehouse"] == "Sud"]
        if south_points.empty:
            break
        new_lat, new_lon = weighted_centroid(south_points)
        delta = abs(new_lat - float(south["lat"])) + abs(new_lon - float(south["lon"]))
        south = {"name": "Entrepot Sud - Barycentre mathematique", "lat": new_lat, "lon": new_lon}
        if previous_assignment == current_assignment or delta < 1e-6:
            break
        previous_assignment = current_assignment

    final_assigned = assign_points(points_df, south, north_ratio_pct)
    return final_assigned, south


def summarize_results(points_df: pd.DataFrame, stores_df: pd.DataFrame, south_warehouse: Warehouse) -> SimulationResult:
    point_to_assignment: Dict[int, str] = {}
    point_to_distance: Dict[int, float] = {}
    for row in points_df.itertuples(index=False):
        for row_id in row.store_row_ids:
            point_to_assignment[int(row_id)] = row.assigned_warehouse
            point_to_distance[int(row_id)] = float(row.distance_km)

    export_df = stores_df.copy()
    export_df["Attribution entrepot"] = export_df["row_id"].map(point_to_assignment)
    export_df["Mode livraison simulation"] = export_df[SCA_COL].map(
        points_df.groupby(SCA_COL)["delivery_mode"].first().to_dict()
    )
    export_df["Distance km vers entrepot attribue"] = export_df["row_id"].map(point_to_distance)

    store_delivery_mask = export_df["Mode livraison simulation"].isin(
        [DELIVERY_STORE, DELIVERY_STORE_SCA_BARYCENTER]
    )
    north_store_mask = store_delivery_mask & (export_df["Attribution entrepot"] == "Nord")
    south_store_mask = store_delivery_mask & (export_df["Attribution entrepot"] == "Sud")

    if north_store_mask.any():
        export_df.loc[north_store_mask, "Distance km vers entrepot attribue"] = haversine_km(
            export_df.loc[north_store_mask, "Latitude"].to_numpy(),
            export_df.loc[north_store_mask, "Longitude"].to_numpy(),
            float(NORD_WAREHOUSE["lat"]),
            float(NORD_WAREHOUSE["lon"]),
        )
    if south_store_mask.any():
        export_df.loc[south_store_mask, "Distance km vers entrepot attribue"] = haversine_km(
            export_df.loc[south_store_mask, "Latitude"].to_numpy(),
            export_df.loc[south_store_mask, "Longitude"].to_numpy(),
            float(south_warehouse["lat"]),
            float(south_warehouse["lon"]),
        )

    export_df["_distance_ponderee"] = export_df["Distance km vers entrepot attribue"] * export_df[WEIGHT_COL]

    warehouse_summary = (
        export_df.groupby("Attribution entrepot")
        .agg(nb_magasins=(STORE_ID_COL, "count"), poids_total=(WEIGHT_COL, "sum"))
        .reset_index()
        .rename(columns={"Attribution entrepot": "Entrepot"})
    )

    distances_summary = (
        export_df.groupby("Attribution entrepot")
        .agg(
            distance_ponderee=("_distance_ponderee", "sum"),
            distance_moyenne_km=("Distance km vers entrepot attribue", "mean"),
        )
        .reset_index()
        .rename(columns={"Attribution entrepot": "Entrepot"})
    )

    total_weighted_distance = float(export_df["_distance_ponderee"].sum())
    export_df = export_df.drop(columns=["_distance_ponderee"])
    return SimulationResult(
        points_df=points_df,
        export_df=export_df,
        warehouse_summary=warehouse_summary,
        distances_summary=distances_summary,
        south_warehouse=south_warehouse,
        total_weighted_distance=total_weighted_distance,
    )


def assigned_store_count(result: SimulationResult, warehouse: str) -> int:
    rows = result.warehouse_summary.loc[result.warehouse_summary["Entrepot"] == warehouse, "nb_magasins"]
    return 0 if rows.empty else int(rows.sum())


def build_visible_points_df(result: SimulationResult) -> pd.DataFrame:
    store_modes = [DELIVERY_STORE, DELIVERY_STORE_SCA_BARYCENTER]
    store_rows = result.export_df.loc[
        result.export_df["Mode livraison simulation"].isin(store_modes)
    ].copy()
    visible_records: List[dict] = []

    for row in store_rows.to_dict(orient="records"):
        visible_records.append(
            {
                "tooltip_name": row["Nom affichage"],
                "display_name": row["Nom affichage"],
                "display_type": "Magasin",
                SCA_COL: row[SCA_COL],
                "delivery_mode": row["Mode livraison simulation"],
                "assigned_warehouse": row["Attribution entrepot"],
                "warehouse_label": row["Attribution entrepot"],
                "Latitude": float(row["Latitude"]),
                "Longitude": float(row["Longitude"]),
                WEIGHT_COL: float(row[WEIGHT_COL]),
                "store_count": 1,
                "distance_km": float(row["Distance km vers entrepot attribue"]),
                "weighted_distance": float(row["Distance km vers entrepot attribue"]) * float(row[WEIGHT_COL]),
            }
        )

    central_points = result.points_df.loc[result.points_df["delivery_mode"] == DELIVERY_CENTRAL]
    for row in central_points.to_dict(orient="records"):
        visible_records.append(
            {
                "tooltip_name": row["display_name"],
                "display_name": row["display_name"],
                "display_type": "Centrale",
                SCA_COL: row[SCA_COL],
                "delivery_mode": row["delivery_mode"],
                "assigned_warehouse": row["assigned_warehouse"],
                "warehouse_label": row["assigned_warehouse"],
                "Latitude": float(row["Latitude"]),
                "Longitude": float(row["Longitude"]),
                WEIGHT_COL: float(row[WEIGHT_COL]),
                "store_count": int(row["store_count"]),
                "distance_km": float(row["distance_km"]),
                "weighted_distance": float(row["weighted_distance"]),
            }
        )

    return pd.DataFrame(visible_records)


def build_map_df(result: SimulationResult) -> pd.DataFrame:
    palette = {"Nord": [33, 113, 181, 190], "Sud": [231, 111, 81, 190]}
    points = build_visible_points_df(result)
    points["color"] = points["assigned_warehouse"].map(palette)
    points["height"] = np.maximum(points[WEIGHT_COL] * 900000, 3000)
    points["poids_display"] = points[WEIGHT_COL].map(lambda x: f"{float(x):.6f}")

    north_weight = points.loc[points["assigned_warehouse"] == "Nord", WEIGHT_COL].sum()
    south_weight = points.loc[points["assigned_warehouse"] == "Sud", WEIGHT_COL].sum()
    warehouse_rows = [
        {
            "tooltip_name": NORD_WAREHOUSE["name"],
            "display_type": "Entrepot",
            "warehouse_label": "Nord",
            "Latitude": NORD_WAREHOUSE["lat"],
            "Longitude": NORD_WAREHOUSE["lon"],
            WEIGHT_COL: north_weight,
            "poids_display": f"{north_weight:.6f}",
            "store_count": int(points.loc[points["assigned_warehouse"] == "Nord", "store_count"].sum()),
            "color": [8, 81, 156, 240],
            "height": max(north_weight * 900000, 5000),
        },
        {
            "tooltip_name": str(result.south_warehouse["name"]),
            "display_type": "Entrepot",
            "warehouse_label": "Sud",
            "Latitude": float(result.south_warehouse["lat"]),
            "Longitude": float(result.south_warehouse["lon"]),
            WEIGHT_COL: south_weight,
            "poids_display": f"{south_weight:.6f}",
            "store_count": int(points.loc[points["assigned_warehouse"] == "Sud", "store_count"].sum()),
            "color": [178, 56, 39, 240],
            "height": max(south_weight * 900000, 5000),
        },
    ]

    warehouses = pd.DataFrame(warehouse_rows)
    points = points[
        [
            "tooltip_name",
            "display_type",
            "warehouse_label",
            "Latitude",
            "Longitude",
            WEIGHT_COL,
            "poids_display",
            "store_count",
            "color",
            "height",
        ]
    ]
    return pd.concat([points, warehouses], ignore_index=True)


def render_map(result: SimulationResult) -> None:
    map_df = build_map_df(result)
    view_state = pdk.ViewState(
        latitude=float(map_df["Latitude"].mean()),
        longitude=float(map_df["Longitude"].mean()),
        zoom=5.4,
        pitch=45,
        bearing=0,
    )
    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position="[Longitude, Latitude]",
        get_elevation="height",
        elevation_scale=1,
        radius=9000,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<b>{tooltip_name}</b><br/>"
            "Type : {display_type}<br/>"
            "Entrepot : {warehouse_label}<br/>"
            "Poids : {poids_display}<br/>"
            "Nombre de magasins : {store_count}"
        )
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light",
        ),
        use_container_width=True,
    )


def export_to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.drop(columns=["row_id"], errors="ignore").to_excel(writer, index=False, sheet_name="Simulation")
    output.seek(0)
    return output.getvalue()


def run_simulation(
    stores_df: pd.DataFrame,
    south_choice: str,
    north_ratio_pct: float,
    delivery_modes: Dict[str, str],
) -> SimulationResult:
    points_df = build_demand_points(stores_df, delivery_modes)
    normalized_choice = south_choice.lower()
    if normalized_choice.startswith("barycentrage"):
        assigned, south_warehouse = compute_barycentric_solution(points_df, north_ratio_pct)
    else:
        matched_key = next((key for key in FIXED_SOUTH_OPTIONS if key.lower() == normalized_choice), south_choice)
        south_warehouse = FIXED_SOUTH_OPTIONS[matched_key]
        assigned = assign_points(points_df, south_warehouse, north_ratio_pct)
    return summarize_results(assigned, stores_df, south_warehouse)


def main() -> None:
    st.title("Simulation transport entrepots Leclerc - vGroup")
    st.caption(
        "Les fichiers Excel sont lus directement dans le dossier du projet. "
        "Les lignes GIE MASCAREIGNE, GIE GIELEC et SODIPLEC sont exclues de la simulation."
    )

    try:
        stores_df, stores_path = load_store_data()
        recap_df, recap_path = load_recap_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    sca_list = sorted(stores_df[SCA_COL].unique().tolist())

    with st.expander("Sources de donnees et hypotheses", expanded=False):
        st.write(f"Fichier magasins : `{stores_path.name}`")
        st.write(f"Fichier centrales : `{recap_path.name}`")
        st.write(f"Nombre de magasins simules apres exclusions : `{len(stores_df)}`")
        st.write(
            "Les coordonnees des centrales sont renseignees dans le code a partir de l'onglet "
            "`recap`, avec un positionnement manuel sur les villes/adresses principales."
        )
        if recap_df.shape[1] >= 7:
            st.dataframe(recap_df.iloc[:, [0, 3, 4, 5, 6]], use_container_width=True, hide_index=True)
        else:
            st.dataframe(recap_df, use_container_width=True, hide_index=True)

    with st.form("simulation_form"):
        left, right = st.columns([1, 1])
        with left:
            south_choice = st.selectbox(
                "Localisation de l'entrepot Sud",
                options=["Toulouse", "Avignon", "Barycentrage mathematique"],
                index=0,
            )
            north_ratio_pct = st.slider(
                "Pourcentage cible de magasins rattaches a l'entrepot Nord",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
            )
        with right:
            st.markdown("**Mode de livraison par SCA regionale**")
            delivery_modes: Dict[str, str] = {}
            col_a, col_b = st.columns(2)
            for idx, sca in enumerate(sca_list):
                target_col = col_a if idx % 2 == 0 else col_b
                options = DELIVERY_OPTIONS if sca in CENTRALS else (DELIVERY_STORE, DELIVERY_STORE_SCA_BARYCENTER)
                with target_col:
                    delivery_modes[sca] = st.selectbox(
                        f"{sca}",
                        options,
                        index=1 if sca in CENTRALS else 0,
                        key=f"delivery_mode_{sca}",
                    )
        launch = st.form_submit_button("Lancer la simulation", type="primary")

    if not launch:
        st.info("Selectionnez les parametres puis cliquez sur `Lancer la simulation`.")
        return

    result = run_simulation(stores_df, south_choice, north_ratio_pct, delivery_modes)

    kpi_1, kpi_2, kpi_3 = st.columns(3)
    kpi_1.metric("Distance ponderee totale", f"{result.total_weighted_distance:,.2f}".replace(",", " "))
    kpi_2.metric("Magasins rattaches au Nord", str(assigned_store_count(result, "Nord")))
    kpi_3.metric(
        "Position Sud simulee",
        f"{float(result.south_warehouse['lat']):.4f}, {float(result.south_warehouse['lon']):.4f}",
    )

    st.subheader("Carte d'attribution")
    render_map(result)

    info_left, info_right = st.columns([1, 1])
    with info_left:
        st.subheader("Synthese par entrepot")
        display_summary = result.warehouse_summary.copy()
        display_summary["poids_total"] = display_summary["poids_total"].map(lambda x: round(float(x), 6))
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
    with info_right:
        st.subheader("Distances ponderees")
        display_distances = result.distances_summary.copy()
        display_distances["distance_ponderee"] = display_distances["distance_ponderee"].map(lambda x: round(float(x), 4))
        display_distances["distance_moyenne_km"] = display_distances["distance_moyenne_km"].map(lambda x: round(float(x), 2))
        st.dataframe(display_distances, use_container_width=True, hide_index=True)

    st.subheader("Points simules")
    display_points = build_visible_points_df(result)[
        [
            "display_name",
            "display_type",
            SCA_COL,
            "delivery_mode",
            "assigned_warehouse",
            "store_count",
            WEIGHT_COL,
            "distance_km",
            "weighted_distance",
        ]
    ].copy()
    display_points[WEIGHT_COL] = display_points[WEIGHT_COL].map(lambda x: round(float(x), 6))
    display_points["distance_km"] = display_points["distance_km"].map(lambda x: round(float(x), 1))
    display_points["weighted_distance"] = display_points["weighted_distance"].map(lambda x: round(float(x), 4))
    st.dataframe(display_points, use_container_width=True, hide_index=True)

    export_bytes = export_to_excel(result.export_df)
    st.download_button(
        "Exporter le fichier Excel avec l'attribution entrepot",
        data=export_bytes,
        file_name="Liste_magasins_revises_simulation_entrepots.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
