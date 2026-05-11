from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Resultats_courbes_transport"
OUTPUT_DIR.mkdir(exist_ok=True)

EXCLUDED_SCAS = {"GIE MASCAREIGNE", "GIE GIELEC", "SODIPLEC"}
WEIGHT_COL = "Poids - CA"
SCA_COL = "Libelle SCA"
STORE_ID_COL = "Identifiant fictif"
Warehouse = Dict[str, Union[float, str]]

NORD_WAREHOUSE: Warehouse = {
    "name": "Entrepot Nord - Ressons-sur-Matz",
    "lat": 49.5380,
    "lon": 2.7751,
}
SOUTH_OPTIONS: Dict[str, Warehouse] = {
    "Toulouse": {"name": "Entrepot Sud - Toulouse", "lat": 43.6045, "lon": 1.4440},
    "Avignon": {"name": "Entrepot Sud - Avignon", "lat": 43.9493, "lon": 4.8055},
    "Barycentre mathematique": {"name": "Entrepot Sud - Barycentre mathematique"},
}

DELIVERY_CENTRAL = "Livraison centrale"
DELIVERY_STORE = "Livraison magasin"
DELIVERY_STORE_SCA_BARYCENTER = "Livraison magasin barycentree SCA"
DELIVERY_OPTIONS = (DELIVERY_CENTRAL, DELIVERY_STORE, DELIVERY_STORE_SCA_BARYCENTER)

TOTAL_VOLUME = 147725 + 290365
FIXED_COST_PER_DELIVERY = 10
VARIABLE_COST_PER_KM = 0.067

# ---------------------------------------------------------------------------
# Parametres a editer avant execution
# ---------------------------------------------------------------------------
# Mode applique aux SCA non listees ci-dessous.
DEFAULT_DELIVERY_MODE = DELIVERY_STORE

# Renseigner ici les SCA a traiter en livraison a leur centrale regionale.
CENTRAL_DELIVERY_SCAS = [
    # "SCADIF",
]

# Renseigner ici les SCA dont l'affectation Nord/Sud doit etre decidee par
# un seul barycentre pondere, tout en calculant le cout final aux magasins.
SCA_BARYCENTER_DELIVERY_SCAS = [
    # "SCACENTRE",
]

# Renseigner ici les SCA forcees en livraison magasin individuelle.
STORE_DELIVERY_SCAS = [
    # "SOCAMIL",
]

# Facultatif : forcer certaines SCA dans un entrepot donne sur toute la courbe.
# Valeurs attendues : "Nord" ou "Sud".
FORCED_WAREHOUSE_BY_SCA: Dict[str, str] = {
    # "SCADIF": "Nord",
}


CENTRALS: Dict[str, Dict[str, Union[str, float]]] = {
    "LECASUD": {"label": "LECASUD - Le Luc", "lat": 43.3948, "lon": 6.3124},
    "SCACENTRE": {"label": "SCACENTRE - Yzeure", "lat": 46.5657, "lon": 3.3546},
    "SCACHAP": {"label": "SCACHAP - Ruffec", "lat": 46.0307, "lon": 0.2003},
    "SCADIF": {"label": "SCADIF - Reau", "lat": 48.5883, "lon": 2.6274},
    "SCALANDES": {"label": "SCALANDES - Mont-de-Marsan", "lat": 43.8921, "lon": -0.5000},
    "SCANORMANDE": {"label": "SCANORMANDE - Lisieux", "lat": 49.1460, "lon": 0.2293},
    "SCAOUEST": {"label": "SCAOUEST - Saint-Etienne-de-Montluc", "lat": 47.2743, "lon": -1.7804},
    "SCAPALSACE": {"label": "SCAPALSACE - Colmar", "lat": 48.0790, "lon": 7.3550},
    "SCAPARTOIS": {"label": "SCAPARTOIS - Tilloy-les-Mofflaines", "lat": 50.2827, "lon": 2.8223},
    "SCAPEST": {"label": "SCAPEST - Saint-Martin-sur-le-Pre", "lat": 48.9724, "lon": 4.3630},
    "SCAPNOR": {"label": "SCAPNOR - Bruyeres-sur-Oise", "lat": 49.1540, "lon": 2.3264},
    "SCARMOR": {"label": "SCARMOR - Landerneau", "lat": 48.4516, "lon": -4.2517},
    "SCASO": {"label": "SCASO - Cestas", "lat": 44.7452, "lon": -0.6813},
    "SOCAMAINE": {"label": "SOCAMAINE - Champagne", "lat": 48.0224, "lon": 0.3320},
    "SOCAMIL": {"label": "SOCAMIL - Castelnaudary", "lat": 43.3189, "lon": 1.9539},
    "SOCARA": {"label": "SOCARA - Villette-d'Anthon", "lat": 45.7941, "lon": 5.1170},
}


def validate_configuration() -> None:
    configured = CENTRAL_DELIVERY_SCAS + SCA_BARYCENTER_DELIVERY_SCAS + STORE_DELIVERY_SCAS
    duplicates = sorted({sca for sca in configured if configured.count(sca) > 1})
    if duplicates:
        raise ValueError(f"SCA presentes dans plusieurs listes de livraison : {', '.join(duplicates)}")

    invalid_forced = sorted(
        sca for sca, warehouse in FORCED_WAREHOUSE_BY_SCA.items() if warehouse not in {"Nord", "Sud"}
    )
    if invalid_forced:
        raise ValueError(f"Forcage entrepot invalide pour : {', '.join(invalid_forced)}")


def delivery_modes_for_scas(sca_list: Iterable[str]) -> Dict[str, str]:
    delivery_modes = {sca: DEFAULT_DELIVERY_MODE for sca in sca_list}
    for sca in CENTRAL_DELIVERY_SCAS:
        delivery_modes[sca] = DELIVERY_CENTRAL
    for sca in SCA_BARYCENTER_DELIVERY_SCAS:
        delivery_modes[sca] = DELIVERY_STORE_SCA_BARYCENTER
    for sca in STORE_DELIVERY_SCAS:
        delivery_modes[sca] = DELIVERY_STORE
    return delivery_modes


def find_excel_path(prefix: str) -> Path:
    matches = sorted(BASE_DIR.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"Impossible de trouver un fichier commencant par {prefix!r}.")
    return matches[0]


def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    normalized = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


def load_store_data() -> pd.DataFrame:
    path = find_excel_path("Liste_magasins")
    df = pd.read_excel(path, sheet_name="Sheet1").copy()
    required_columns = {SCA_COL, STORE_ID_COL, "Latitude", "Longitude", WEIGHT_COL, "Raison sociale", "Ville"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {', '.join(missing_columns)}")

    df["row_id"] = np.arange(len(df))
    filtered = df.loc[~df[SCA_COL].isin(EXCLUDED_SCAS)].copy()
    filtered["Latitude"] = parse_numeric_series(filtered["Latitude"])
    filtered["Longitude"] = parse_numeric_series(filtered["Longitude"])
    filtered[WEIGHT_COL] = parse_numeric_series(filtered[WEIGHT_COL])
    filtered = filtered.dropna(subset=["Latitude", "Longitude", WEIGHT_COL]).reset_index(drop=True)
    filtered["Nom affichage"] = filtered["Raison sociale"].fillna(filtered["Ville"].astype(str))
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


def transport_cost(distance_km: np.ndarray, weights: np.ndarray) -> np.ndarray:
    volume = weights * TOTAL_VOLUME
    return (FIXED_COST_PER_DELIVERY + VARIABLE_COST_PER_KM * distance_km) * volume


def weighted_centroid(df: pd.DataFrame) -> Tuple[float, float]:
    weights = df[WEIGHT_COL].to_numpy()
    lat = np.average(df["Latitude"].to_numpy(), weights=weights)
    lon = np.average(df["Longitude"].to_numpy(), weights=weights)
    return float(lat), float(lon)


def build_demand_points(stores_df: pd.DataFrame, delivery_modes: Dict[str, str]) -> pd.DataFrame:
    records: List[dict] = []
    for sca, group in stores_df.groupby(SCA_COL, sort=True):
        mode = delivery_modes.get(sca, DEFAULT_DELIVERY_MODE)
        if mode == DELIVERY_CENTRAL:
            if sca not in CENTRALS:
                mode = DELIVERY_STORE
            else:
                central = CENTRALS[sca]
                records.append(
                    {
                        "point_id": f"CENTRALE::{sca}",
                        "display_name": central["label"],
                        "display_type": "Centrale",
                        SCA_COL: sca,
                        "Latitude": float(central["lat"]),
                        "Longitude": float(central["lon"]),
                        WEIGHT_COL: group[WEIGHT_COL].sum(),
                        "store_count": int(len(group)),
                        "delivery_mode": mode,
                        "store_row_ids": tuple(group["row_id"].tolist()),
                    }
                )
                continue

        if mode == DELIVERY_STORE_SCA_BARYCENTER:
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
                        "delivery_mode": DELIVERY_STORE,
                        "store_row_ids": (int(row["row_id"]),),
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
    forced = points_df[SCA_COL].map(FORCED_WAREHOUSE_BY_SCA)
    forced_north = forced == "Nord"
    forced_south = forced == "Sud"
    free_mask = ~(forced_north | forced_south)

    assignment = np.array(forced_north.to_numpy(dtype=bool), dtype = bool, copy = True)
    forced_north_stores = int(points_df.loc[forced_north, "store_count"].sum())
    free_points = points_df.loc[free_mask].copy()
    free_indices = points_df.index[free_mask].to_numpy()
    free_target = north_target - forced_north_stores
    free_total_stores = int(free_points["store_count"].sum())

    if free_points.empty or free_target <= 0:
        return assignment
    if free_target >= free_total_stores:
        assignment[free_indices] = True
        return assignment

    projections = compute_axis_projection(
        free_points["Latitude"].to_numpy(),
        free_points["Longitude"].to_numpy(),
        float(NORD_WAREHOUSE["lat"]),
        float(NORD_WAREHOUSE["lon"]),
        float(south_warehouse["lat"]),
        float(south_warehouse["lon"]),
    )

    ordered = free_points.copy()
    ordered["projection"] = projections
    ordered["north_cost"] = north_cost[free_indices]
    ordered["south_cost"] = south_cost[free_indices]
    ordered = ordered.sort_values("projection", ascending=False).reset_index()

    cum_store_count = ordered["store_count"].cumsum().to_numpy()
    cum_north_cost = ordered["north_cost"].cumsum().to_numpy()
    south_suffix = ordered["south_cost"][::-1].cumsum()[::-1].to_numpy()
    total_south_cost = float(ordered["south_cost"].sum())

    candidate_prefixes = {0, len(ordered)}
    for idx, store_count in enumerate(cum_store_count):
        if int(store_count) <= free_target:
            candidate_prefixes.add(idx + 1)
        else:
            candidate_prefixes.add(idx + 1)
            candidate_prefixes.add(idx)
            break

    best_prefix = 0
    best_gap = free_total_stores
    best_total_cost = total_south_cost
    for prefix in sorted(p for p in candidate_prefixes if 0 <= p <= len(ordered)):
        assigned_store_count = 0 if prefix == 0 else int(cum_store_count[prefix - 1])
        gap = abs(assigned_store_count - free_target)
        candidate_total = 0.0 if prefix == 0 else float(cum_north_cost[prefix - 1])
        if prefix < len(ordered):
            candidate_total += float(south_suffix[prefix])
        if gap < best_gap or (gap == best_gap and candidate_total < best_total_cost - 1e-12):
            best_gap = gap
            best_total_cost = candidate_total
            best_prefix = prefix

    if best_prefix > 0:
        assignment[ordered.loc[: best_prefix - 1, "index"].to_numpy(dtype=int)] = True
    return assignment


def assign_points(points_df: pd.DataFrame, south_warehouse: Warehouse, north_ratio_pct: int) -> pd.DataFrame:
    total_stores = int(points_df["store_count"].sum())
    north_target = int(round(total_stores * north_ratio_pct / 100.0))

    lat = points_df["Latitude"].to_numpy()
    lon = points_df["Longitude"].to_numpy()
    weights = points_df[WEIGHT_COL].to_numpy()

    north_dist = haversine_km(lat, lon, float(NORD_WAREHOUSE["lat"]), float(NORD_WAREHOUSE["lon"]))
    south_dist = haversine_km(lat, lon, float(south_warehouse["lat"]), float(south_warehouse["lon"]))
    north_cost = transport_cost(north_dist, weights)
    south_cost = transport_cost(south_dist, weights)
    north_mask = solve_north_assignment(points_df, north_cost, south_cost, north_target, south_warehouse)

    assigned = points_df.copy()
    assigned["assigned_warehouse"] = np.where(north_mask, "Nord", "Sud")
    assigned["distance_km"] = np.where(north_mask, north_dist, south_dist)
    assigned["transport_cost"] = np.where(north_mask, north_cost, south_cost)
    assigned["north_cost"] = north_cost
    assigned["south_cost"] = south_cost
    return assigned


def compute_barycentric_solution(points_df: pd.DataFrame, north_ratio_pct: int) -> Tuple[pd.DataFrame, Warehouse]:
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

    return assign_points(points_df, south, north_ratio_pct), south


def summarize_store_costs(assigned_points: pd.DataFrame, stores_df: pd.DataFrame, south_warehouse: Warehouse) -> pd.DataFrame:
    point_to_assignment: Dict[int, str] = {}
    point_to_mode: Dict[int, str] = {}
    point_to_cost: Dict[int, float] = {}
    for row in assigned_points.itertuples(index=False):
        cost_per_store = float(row.transport_cost) / max(int(row.store_count), 1)
        for row_id in row.store_row_ids:
            point_to_assignment[int(row_id)] = row.assigned_warehouse
            point_to_mode[int(row_id)] = row.delivery_mode
            point_to_cost[int(row_id)] = cost_per_store

    export_df = stores_df.copy()
    export_df["assigned_warehouse"] = export_df["row_id"].map(point_to_assignment)
    export_df["delivery_mode"] = export_df["row_id"].map(point_to_mode)
    export_df["transport_cost"] = export_df["row_id"].map(point_to_cost)

    store_delivery_mask = export_df["delivery_mode"].isin([DELIVERY_STORE, DELIVERY_STORE_SCA_BARYCENTER])
    north_store_mask = store_delivery_mask & (export_df["assigned_warehouse"] == "Nord")
    south_store_mask = store_delivery_mask & (export_df["assigned_warehouse"] == "Sud")

    if north_store_mask.any():
        north_dist = haversine_km(
            export_df.loc[north_store_mask, "Latitude"].to_numpy(),
            export_df.loc[north_store_mask, "Longitude"].to_numpy(),
            float(NORD_WAREHOUSE["lat"]),
            float(NORD_WAREHOUSE["lon"]),
        )
        export_df.loc[north_store_mask, "transport_cost"] = transport_cost(
            north_dist,
            export_df.loc[north_store_mask, WEIGHT_COL].to_numpy(),
        )
    if south_store_mask.any():
        south_dist = haversine_km(
            export_df.loc[south_store_mask, "Latitude"].to_numpy(),
            export_df.loc[south_store_mask, "Longitude"].to_numpy(),
            float(south_warehouse["lat"]),
            float(south_warehouse["lon"]),
        )
        export_df.loc[south_store_mask, "transport_cost"] = transport_cost(
            south_dist,
            export_df.loc[south_store_mask, WEIGHT_COL].to_numpy(),
        )

    return export_df


def run_scenario(stores_df: pd.DataFrame, delivery_modes: Dict[str, str], south_choice: str) -> pd.DataFrame:
    demand_points = build_demand_points(stores_df, delivery_modes)
    records: List[dict] = []

    for pct in range(101):
        if south_choice == "Barycentre mathematique":
            assigned_points, south_warehouse = compute_barycentric_solution(demand_points, pct)
        else:
            south_warehouse = SOUTH_OPTIONS[south_choice]
            assigned_points = assign_points(demand_points, south_warehouse, pct)

        store_costs = summarize_store_costs(assigned_points, stores_df, south_warehouse)
        north_mask = store_costs["assigned_warehouse"] == "Nord"
        south_mask = ~north_mask

        records.append(
            {
                "Pourcentage magasins Nord": pct,
                "Nombre magasins Nord": int(north_mask.sum()),
                "Nombre magasins Sud": int(south_mask.sum()),
                "Poids Nord": float(store_costs.loc[north_mask, WEIGHT_COL].sum()),
                "Poids Sud": float(store_costs.loc[south_mask, WEIGHT_COL].sum()),
                "Cout transport Nord": float(store_costs.loc[north_mask, "transport_cost"].sum()),
                "Cout transport Sud": float(store_costs.loc[south_mask, "transport_cost"].sum()),
                "Cout transport total": float(store_costs["transport_cost"].sum()),
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
    plt.plot(results_df["Pourcentage magasins Nord"], results_df["Cout transport total"] / 1e6, linewidth=2.5)
    plt.title(f"Cout total de transport selon % Nord\nEntrepot Sud : {scenario_name}")
    plt.xlabel("Pourcentage de magasins rattaches a Ressons-sur-Matz")
    plt.ylabel("Cout total de transport (MEUR)")
    plt.grid(True, alpha=0.25)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_combined_chart(results_by_scenario: Dict[str, pd.DataFrame]) -> Path:
    output_path = OUTPUT_DIR / "cout_transport_comparatif_3_scenarios.png"

    plt.figure(figsize=(11, 7))
    for scenario_name, results_df in results_by_scenario.items():
        plt.plot(
            results_df["Pourcentage magasins Nord"],
            results_df["Cout transport total"] / 1e6,
            linewidth=2.5,
            label=scenario_name,
        )

    plt.title("Cout total de transport selon % Nord\nComparatif des 3 localisations Sud")
    plt.xlabel("Pourcentage de magasins rattaches a Ressons-sur-Matz")
    plt.ylabel("Cout total de transport (MEUR)")
    plt.grid(True, alpha=0.25)
    plt.xlim(0, 100)
    plt.legend(title="Entrepot Sud")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    validate_configuration()
    stores_df = load_store_data()
    delivery_modes = delivery_modes_for_scas(sorted(stores_df[SCA_COL].unique().tolist()))
    print(f"Magasins pris en compte : {len(stores_df)}")
    print("Modes de livraison :")
    for mode in DELIVERY_OPTIONS:
        scas = sorted(sca for sca, selected_mode in delivery_modes.items() if selected_mode == mode)
        print(f"- {mode} : {len(scas)} SCA")

    generated_files: List[Path] = []
    results_by_scenario: Dict[str, pd.DataFrame] = {}
    for south_choice in SOUTH_OPTIONS:
        print(f"Calcul en cours : {south_choice}")
        results_df = run_scenario(stores_df, delivery_modes, south_choice)
        results_by_scenario[south_choice] = results_df
        excel_path = save_excel(results_df, south_choice)
        chart_path = save_chart(results_df, south_choice)
        generated_files.extend([excel_path, chart_path])
        print(f"  Excel : {excel_path.name}")
        print(f"  Courbe : {chart_path.name}")

    combined_chart_path = save_combined_chart(results_by_scenario)
    generated_files.append(combined_chart_path)
    print(f"  Courbe comparative : {combined_chart_path.name}")

    print("\nFichiers generes :")
    for path in generated_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
