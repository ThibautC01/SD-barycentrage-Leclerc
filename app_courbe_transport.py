from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "Resultats_courbes_transport"
OUTPUT_DIR.mkdir(exist_ok=True)

EXCLUDED_SCAS = {"GIE MASCAREIGNE", "GIE GIELEC", "SODIPLEC"}
WEIGHT_COL = "Poids - CA"
SCA_COL = "Libelle SCA"

NORD_WAREHOUSE = {
    "name": "Entrepôt Nord - Ressons-sur-Matz",
    "lat": 49.5380,
    "lon": 2.7751,
}
SOUTH_OPTIONS = {
    "Toulouse": {"name": "Entrepôt Sud - Toulouse", "lat": 43.6045, "lon": 1.4440},
    "Avignon": {"name": "Entrepôt Sud - Avignon", "lat": 43.9493, "lon": 4.8055},
    "Barycentre mathematique": {"name": "Entrepôt Sud - Barycentre mathématique"},
}

TOTAL_VOLUME = 147725 + 290365 # volume palettes estimé à 2030 (promo + permanent)

def find_excel_path(prefix: str) -> Path:
    matches = sorted(BASE_DIR.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"Impossible de trouver un fichier commençant par {prefix!r}.")
    return matches[0]


def load_store_data() -> pd.DataFrame:
    path = find_excel_path("Liste_magasins")
    df = pd.read_excel(path, sheet_name="Sheet1")
    filtered = df.loc[~df[SCA_COL].isin(EXCLUDED_SCAS)].copy()
    filtered["Latitude"] = pd.to_numeric(filtered["Latitude"], errors="coerce")
    filtered["Longitude"] = pd.to_numeric(filtered["Longitude"], errors="coerce")
    filtered[WEIGHT_COL] = pd.to_numeric(filtered[WEIGHT_COL], errors="coerce")
    filtered = filtered.dropna(subset=["Latitude", "Longitude", WEIGHT_COL]).reset_index(drop=True)
    return filtered


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
    south_warehouse: Dict[str, float | str],
) -> np.ndarray:
    total_stores = len(points_df)

    if north_target <= 0:
        return np.zeros(len(points_df), dtype=bool)
    if north_target >= total_stores:
        return np.ones(len(points_df), dtype=bool)

    projections = compute_axis_projection(
        points_df["Latitude"].to_numpy(),
        points_df["Longitude"].to_numpy(),
        NORD_WAREHOUSE["lat"],
        NORD_WAREHOUSE["lon"],
        float(south_warehouse["lat"]),
        float(south_warehouse["lon"]),
    )

    ordered = points_df.copy()
    ordered["projection"] = projections
    ordered["north_cost"] = north_cost
    ordered["south_cost"] = south_cost
    ordered = ordered.sort_values("projection", ascending=False).reset_index()

    cum_north_cost = ordered["north_cost"].cumsum().to_numpy()
    south_suffix = ordered["south_cost"][::-1].cumsum()[::-1].to_numpy()
    total_south_cost = float(ordered["south_cost"].sum())

    candidate_prefixes = {0, len(ordered)}
    for prefix in range(len(ordered)):
        if prefix + 1 == north_target:
            candidate_prefixes.add(prefix + 1)
        elif prefix + 1 < north_target:
            candidate_prefixes.add(prefix + 1)
        elif prefix + 1 > north_target:
            candidate_prefixes.add(prefix + 1)
            candidate_prefixes.add(prefix)
            break

    best_prefix = 0
    best_gap = total_stores
    best_total_cost = total_south_cost

    for prefix in sorted(p for p in candidate_prefixes if 0 <= p <= len(ordered)):
        assigned_store_count = prefix
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

def assign_points(points_df: pd.DataFrame, south_warehouse: Dict[str, float | str], north_ratio_pct: int) -> pd.DataFrame:
    north_target = int(round(len(points_df) * north_ratio_pct / 100.0))

    lat = points_df["Latitude"].to_numpy()
    lon = points_df["Longitude"].to_numpy()
    weights = points_df[WEIGHT_COL].to_numpy()

    # Distances
    north_dist = haversine_km(lat, lon, NORD_WAREHOUSE["lat"], NORD_WAREHOUSE["lon"])
    south_dist = haversine_km(lat, lon, float(south_warehouse["lat"]), float(south_warehouse["lon"]))

    # Volume par magasin (proportionnel au poids)
    volume_magasin = weights * TOTAL_VOLUME

    # Coût transport
    north_cost = (10 + 0.067 * north_dist) * volume_magasin
    south_cost = (10 + 0.067 * south_dist) * volume_magasin

    north_mask = solve_north_assignment(points_df, north_cost, south_cost, north_target, south_warehouse)

    assigned = points_df.copy()
    assigned["assigned_warehouse"] = np.where(north_mask, "Nord", "Sud")
    assigned["distance_km"] = np.where(north_mask, north_dist, south_dist)
    assigned["transport_cost"] = np.where(north_mask, north_cost, south_cost)

    return assigned


def compute_barycentric_solution(points_df: pd.DataFrame, north_ratio_pct: int) -> Tuple[pd.DataFrame, Dict[str, float | str]]:
    lat, lon = weighted_centroid(points_df)
    south = {"name": "Entrepôt Sud - Barycentre mathématique", "lat": lat, "lon": lon}
    previous_assignment = None

    for _ in range(30):
        assigned = assign_points(points_df, south, north_ratio_pct)
        current_assignment = assigned["assigned_warehouse"].tolist()
        south_points = assigned.loc[assigned["assigned_warehouse"] == "Sud"]
        if south_points.empty:
            break
        new_lat, new_lon = weighted_centroid(south_points)
        delta = abs(new_lat - float(south["lat"])) + abs(new_lon - float(south["lon"]))
        south = {"name": "Entrepôt Sud - Barycentre mathématique", "lat": new_lat, "lon": new_lon}
        if previous_assignment == current_assignment or delta < 1e-6:
            break
        previous_assignment = current_assignment

    return assign_points(points_df, south, north_ratio_pct), south


def run_scenario(stores_df: pd.DataFrame, south_choice: str) -> pd.DataFrame:
    records: List[dict] = []

    for pct in range(101):
        if south_choice == "Barycentre mathematique":
            assigned, south_warehouse = compute_barycentric_solution(stores_df, pct)
        else:
            south_warehouse = SOUTH_OPTIONS[south_choice]
            assigned = assign_points(stores_df, south_warehouse, pct)

        north_mask = assigned["assigned_warehouse"] == "Nord"
        south_mask = ~north_mask

        records.append(
            {
                "Pourcentage magasins Nord": pct,
                "Nombre magasins Nord": int(north_mask.sum()),
                "Nombre magasins Sud": int(south_mask.sum()),
                "Poids Nord": float(assigned.loc[north_mask, WEIGHT_COL].sum()),
                "Poids Sud": float(assigned.loc[south_mask, WEIGHT_COL].sum()),
                "Cout transport Nord": float(assigned.loc[north_mask, "transport_cost"].sum()),
                "Cout transport Sud": float(assigned.loc[south_mask, "transport_cost"].sum()),
                "Cout transport total": float(assigned["transport_cost"].sum()),
                "Latitude entrepot Sud": float(south_warehouse["lat"]),
                "Longitude entrepot Sud": float(south_warehouse["lon"]),
            }
        )

    return pd.DataFrame(records)


def save_excel(results_df: pd.DataFrame, scenario_name: str) -> Path:
    safe_name = scenario_name.lower().replace(" ", "_")
    output_path = OUTPUT_DIR / f"cout_transport_{safe_name}.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Courbe")

    return output_path


def save_chart(results_df: pd.DataFrame, scenario_name: str) -> Path:
    safe_name = scenario_name.lower().replace(" ", "_")
    output_path = OUTPUT_DIR / f"cout_transport_{safe_name}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(
        results_df["Pourcentage magasins Nord"],
        results_df["Cout transport total"] / 1e6,
        linewidth=2.5,
    )
    plt.title(f"Coût total de transport selon % Nord\nEntrepôt Sud : {scenario_name}")
    plt.xlabel("Pourcentage de magasins rattachés à Ressons-sur-Matz")
    plt.ylabel("Coût total de transport (M€)")
    plt.grid(True, alpha=0.25)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main() -> None:
    stores_df = load_store_data()
    print(f"Magasins pris en compte : {len(stores_df)}")

    generated_files: List[Path] = []
    for south_choice in SOUTH_OPTIONS:
        print(f"Calcul en cours : {south_choice}")
        results_df = run_scenario(stores_df, south_choice)
        excel_path = save_excel(results_df, south_choice)
        chart_path = save_chart(results_df, south_choice)
        generated_files.extend([excel_path, chart_path])
        print(f"  Excel : {excel_path.name}")
        print(f"  Courbe : {chart_path.name}")

    print("\nFichiers générés :")
    for path in generated_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
