"""
=============================================================================
 INTA AMBIENTE: Plataforma de Ambientación con Sentinel-2 y Google Earth Engine
=============================================================================
 Diseñado y Desarrollado por:
 Lic. Inf. Hernan Elena
 Grupo de Recursos Naturales (RRNN) - INTA EEA Salta
 Proyecto PATCHACO
=============================================================================
"""

import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw, Fullscreen, SideBySideLayers
import json
import datetime
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, Point
import plotly.express as px
import plotly.graph_objects as go

# Constante de protección operativa para no saturar Google Earth Engine
MAX_SUPERFICIE_HA = 10000.0

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="INTA AMBIENTE | Sentinel-2 GEE",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Tipografía y encabezados principales */
    .app-header {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 0.8rem;
    }
    .app-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b4332;
        margin: 0;
        line-height: 1.1;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #40916c;
        margin: 0;
        font-weight: 500;
    }
    .app-badge {
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b4332;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #c8e6c9;
        margin-top: 4px;
    }

    /* Tarjetas de métricas responsivas (evitan desbordes de pantalla) */
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 15px;
    }
    .metric-pill {
        flex: 1 1 200px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2d6a4f;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-pill-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-pill-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-pill-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Leyenda de ambientes con alto contraste */
    .legend-box {
        background: #1e293b;
        color: #f8fafc !important;
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
        border: 1px solid #334155;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 22px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #f8fafc !important;
    }
    .legend-color {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        margin-right: 8px;
        display: inline-block;
        border: 1px solid rgba(0,0,0,0.2);
    }

    /* Botones de descarga */
    .stDownloadButton button {
        width: 100%;
        background-color: #2d6a4f !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1rem !important;
    }
    .stDownloadButton button:hover {
        background-color: #1b4332 !important;
    }

    /* Pie de página institucional */
    .footer-credits {
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.85rem;
        color: #64748b;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. AUTENTICACIÓN GOOGLE EARTH ENGINE (GEE)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_earth_engine():
    """Inicializa Earth Engine usando Service Account Credentials desde st.secrets."""
    try:
        cred_dict = None
        if "gcp_service_account" in st.secrets:
            cred_dict = dict(st.secrets["gcp_service_account"])
        elif "earth_engine" in st.secrets:
            cred_dict = dict(st.secrets["earth_engine"])
        elif "client_email" in st.secrets:
            cred_dict = dict(st.secrets)
        else:
            local_json = "ee-hernanelena-807b3c62c296.json"
            if os.path.exists(local_json):
                with open(local_json, "r") as f:
                    cred_dict = json.load(f)

        if not cred_dict:
            return False, "No se encontraron credenciales en st.secrets ni archivo JSON local."

        client_email = cred_dict.get("client_email")
        project_id = cred_dict.get("project_id", "ee-hernanelena")

        credentials = ee.ServiceAccountCredentials(
            client_email,
            key_data=json.dumps(cred_dict)
        )

        # Inicialización de Earth Engine usando la Cuenta de Servicio
        # Importante: Inicializar directamente con credentials evita el error
        # "Caller does not have required permission to use project" de Google Cloud
        try:
            ee.Initialize(credentials=credentials)
            _ = ee.Number(1).getInfo()
        except Exception:
            ee.Initialize(credentials=credentials, project=project_id)
            _ = ee.Number(1).getInfo()

        return True, "Conexión exitosa con Google Earth Engine."

    except Exception as e:
        return False, f"Error al inicializar Earth Engine: {str(e)}"


# ---------------------------------------------------------------------------
# 3. UTILIDADES DE COORDENADAS (DECIMAL <-> SEXAGESIMAL / DMS)
# ---------------------------------------------------------------------------
def decimal_to_dms(val, is_lat=True):
    """Convierte grados decimales a formato sexagesimal (DMS)."""
    if val is None:
        return ""
    abs_v = abs(val)
    d = int(abs_v)
    m = int((abs_v - d) * 60)
    s = round((abs_v - d - m / 60.0) * 3600.0, 1)
    if is_lat:
        hemi = "S" if val < 0 else "N"
    else:
        hemi = "O" if val < 0 else "E"
    return f"{d}° {m}' {s}\" {hemi}"


def parse_free_coordinate_string(text):
    """Parsea una cadena en decimal o sexagesimal."""
    text = text.strip()
    dec_match = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(dec_match) == 2 and ("°" not in text and "'" not in text):
        try:
            lat = float(dec_match[0])
            lon = float(dec_match[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except Exception:
            pass

    try:
        parts = re.split(r"[,;]", text)
        if len(parts) >= 2:
            lat_str, lon_str = parts[0].strip(), parts[1].strip()

            def extract_dms(s):
                hemi = "N"
                for h in ["S", "N", "O", "W", "E"]:
                    if h in s.upper():
                        hemi = h
                        break
                nums = [float(x) for x in re.findall(r"\d+\.?\d*", s)]
                if len(nums) == 1:
                    v = nums[0]
                elif len(nums) == 2:
                    v = nums[0] + nums[1] / 60.0
                elif len(nums) >= 3:
                    v = nums[0] + nums[1] / 60.0 + nums[2] / 3600.0
                else:
                    return None
                if hemi in ["S", "O", "W"] or "-" in s:
                    v = -abs(v)
                return v

            lat = extract_dms(lat_str)
            lon = extract_dms(lon_str)
            if lat is not None and lon is not None:
                return lat, lon
    except Exception:
        pass

    return None, None


# ---------------------------------------------------------------------------
# 4. PARSERS VECTORIALES (KML & GEOJSON)
# ---------------------------------------------------------------------------
def parse_kml_to_shapely(kml_bytes):
    """Parser XML nativo agnóstico de namespaces XML."""
    try:
        tree = ET.fromstring(kml_bytes)
        polygons = []

        for elem in tree.iter():
            tag = elem.tag.split("}")[-1]
            if tag == "Polygon":
                for sub in elem.iter():
                    sub_tag = sub.tag.split("}")[-1]
                    if sub_tag == "coordinates" and sub.text:
                        raw_coords = sub.text.strip().split()
                        coords = []
                        for pt in raw_coords:
                            parts = [float(x) for x in pt.split(",") if x]
                            if len(parts) >= 2:
                                coords.append((parts[0], parts[1]))
                        if len(coords) >= 3:
                            poly = Polygon(coords)
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if not poly.is_empty:
                                polygons.append(poly)
                        break

        if not polygons:
            for elem in tree.iter():
                tag = elem.tag.split("}")[-1]
                if tag in ["LinearRing", "LineString"]:
                    for sub in elem.iter():
                        sub_tag = sub.tag.split("}")[-1]
                        if sub_tag == "coordinates" and sub.text:
                            raw_coords = sub.text.strip().split()
                            coords = []
                            for pt in raw_coords:
                                parts = [float(x) for x in pt.split(",") if x]
                                if len(parts) >= 2:
                                    coords.append((parts[0], parts[1]))
                            if len(coords) >= 4:
                                poly = Polygon(coords)
                                if not poly.is_valid:
                                    poly = poly.buffer(0)
                                if not poly.is_empty:
                                    polygons.append(poly)
                            break

        if not polygons:
            return None, "No se encontraron polígonos válidos en el archivo KML."

        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons), None
    except Exception as e:
        return None, f"Error al procesar KML: {str(e)}"


def parse_geojson_to_shapely(geojson_data):
    """Convierte GeoJSON dict o str en geometría Shapely."""
    try:
        if isinstance(geojson_data, str):
            data = json.loads(geojson_data)
        elif isinstance(geojson_data, bytes):
            data = json.loads(geojson_data.decode("utf-8"))
        else:
            data = geojson_data

        if data.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
            if not geoms:
                return None, "FeatureCollection sin geometrías válidas."
            return geoms[0] if len(geoms) == 1 else MultiPolygon([g for g in geoms if isinstance(g, Polygon)]), None
        elif data.get("type") == "Feature":
            return shape(data["geometry"]), None
        elif data.get("type") in ["Polygon", "MultiPolygon"]:
            return shape(data), None
        else:
            return None, "Formato GeoJSON no reconocido como polígono."
    except Exception as e:
        return None, f"Error al decodificar GeoJSON: {str(e)}"


def shapely_to_ee_geometry(shapely_geom):
    """Convierte una geometría Shapely a ee.Geometry."""
    geojson_dict = mapping(shapely_geom)
    return ee.Geometry(geojson_dict)


# ---------------------------------------------------------------------------
# 5. MOTOR DE PROCESAMIENTO SENTINEL-2 Y SUITE DE ÍNDICES AGRONÓMICOS
# ---------------------------------------------------------------------------
def mask_s2_clouds_scl(image):
    """Máscara de nubes usando la banda SCL de Sentinel-2 L2A."""
    scl = image.select("SCL")
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    bands = image.select(["B2", "B3", "B4", "B5", "B8", "B11"]).divide(10000.0)
    return bands.updateMask(mask).copyProperties(image, ["system:time_start"])


def get_sentinel2_composite(aoi_ee, start_date, end_date, max_cloud_pct):
    """Filtra Sentinel-2 Harmonized SR y calcula la mediana temporal."""
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_ee)
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )

    image_count = s2_col.size().getInfo()
    if image_count == 0:
        return None, 0

    s2_clean = s2_col.map(mask_s2_clouds_scl)
    composite = s2_clean.median().clip(aoi_ee)
    return composite, image_count


def calculate_agronomic_indices(composite_img):
    """Calcula la suite completa de índices espectrales agronómicos."""
    ndvi = composite_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = composite_img.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndwi = composite_img.normalizedDifference(["B8", "B11"]).rename("NDWI")
    ica = composite_img.expression(
        "0.40 * ndvi + 0.35 * ndre + 0.25 * ndwi",
        {"ndvi": ndvi, "ndre": ndre, "ndwi": ndwi}
    ).rename("ICA")

    return composite_img.addBands([ndvi, ndre, ndwi, ica])


# ---------------------------------------------------------------------------
# 6. ALGORITMOS DE AMBIENTACIÓN (MONOVARIABLE & MULTIVARIADO)
# ---------------------------------------------------------------------------
def classify_zones_quantiles(img_indices, aoi_ee, target_band="NDVI", num_zones=4, smooth=True):
    """Segmentación por Cuantiles/Percentiles de una banda específica."""
    band = img_indices.select(target_band)

    if num_zones == 3:
        percentiles_to_calc = [33, 66]
    else:
        percentiles_to_calc = [25, 50, 75]

    pct_dict = band.reduceRegion(
        reducer=ee.Reducer.percentile(percentiles_to_calc),
        geometry=aoi_ee,
        scale=10,
        maxPixels=1e8
    ).getInfo()

    prefix = f"{target_band}_p"
    if num_zones == 3:
        p33 = pct_dict.get(f"{prefix}33", 0.3)
        p66 = pct_dict.get(f"{prefix}66", 0.6)
        classified = (
            ee.Image(1)
            .where(band.gte(p33), 2)
            .where(band.gte(p66), 3)
            .rename("zone")
            .clip(aoi_ee)
        )
        thresholds = {f"P33 ({target_band})": round(p33, 3), f"P66 ({target_band})": round(p66, 3)}
    else:
        p25 = pct_dict.get(f"{prefix}25", 0.25)
        p50 = pct_dict.get(f"{prefix}50", 0.50)
        p75 = pct_dict.get(f"{prefix}75", 0.75)
        classified = (
            ee.Image(1)
            .where(band.gte(p25), 2)
            .where(band.gte(p50), 3)
            .where(band.gte(p75), 4)
            .rename("zone")
            .clip(aoi_ee)
        )
        thresholds = {
            f"P25 ({target_band})": round(p25, 3),
            f"P50 ({target_band})": round(p50, 3),
            f"P75 ({target_band})": round(p75, 3)
        }

    if smooth:
        classified = classified.focalMode(radius=10, units="meters").rename("zone").clip(aoi_ee)

    return classified, thresholds


def classify_zones_multivariate_kmeans(img_indices, aoi_ee, num_zones=4, smooth=True):
    """Ambientación Multivariada Completa (NDVI + NDRE + NDWI) mediante K-Means."""
    features_stack = img_indices.select(["NDVI", "NDRE", "NDWI"])

    training = features_stack.sample(
        region=aoi_ee,
        scale=10,
        numPixels=1000,
        seed=42
    )

    clusterer = ee.Clusterer.wekaKMeans(num_zones).train(training)
    clustered = features_stack.cluster(clusterer).rename("zone_raw")

    cluster_profiles = []
    for c in range(num_zones):
        mask_c = clustered.eq(c)
        stats = features_stack.updateMask(mask_c).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_ee,
            scale=10,
            maxPixels=1e8
        ).getInfo()

        m_ndvi = stats.get("NDVI", 0.0) or 0.0
        m_ndre = stats.get("NDRE", 0.0) or 0.0
        m_ndwi = stats.get("NDWI", 0.0) or 0.0
        
        score = 0.40 * m_ndvi + 0.35 * m_ndre + 0.25 * m_ndwi
        cluster_profiles.append({
            "raw_id": c,
            "score": score,
            "ndvi": round(m_ndvi, 3),
            "ndre": round(m_ndre, 3),
            "ndwi": round(m_ndwi, 3)
        })

    sorted_profiles = sorted(cluster_profiles, key=lambda x: x["score"])

    classified = ee.Image(0)
    reordered_metrics = {}
    for new_id, p in enumerate(sorted_profiles, start=1):
        classified = classified.where(clustered.eq(p["raw_id"]), new_id)
        reordered_metrics[f"Ambiente {new_id}"] = {
            "NDVI Medio": p["ndvi"],
            "NDRE Medio (Clorofila)": p["ndre"],
            "NDWI Medio (Agua/Suelo)": p["ndwi"],
            "Score Combinado": round(p["score"], 3)
        }

    classified = classified.rename("zone").clip(aoi_ee)

    if smooth:
        classified = classified.focalMode(radius=10, units="meters").rename("zone").clip(aoi_ee)

    return classified, reordered_metrics


def vectorize_zones_to_gdf(classified_img, aoi_ee, num_zones=4):
    """Vectoriza el ráster clasificado a GeoPandas GeoDataFrame con cálculo de hectáreas."""
    vectors = classified_img.reduceToVectors(
        geometry=aoi_ee,
        scale=10,
        geometryType="polygon",
        eightConnected=False,
        labelProperty="zone_id",
        reducer=ee.Reducer.countEvery(),
        maxPixels=1e8
    )

    fc_info = vectors.getInfo()
    features = fc_info.get("features", [])
    if not features:
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf["zone_id"] = gdf["zone_id"].astype(int)

    if num_zones == 3:
        labels_map = {
            1: "Ambiente 1 (Vigor / Actividad Baja)",
            2: "Ambiente 2 (Vigor / Actividad Media)",
            3: "Ambiente 3 (Vigor / Actividad Alta)"
        }
        color_hex_map = {1: "#d73027", 2: "#fee08b", 3: "#1a9850"}
    else:
        labels_map = {
            1: "Ambiente 1 (Vigor / Actividad Baja)",
            2: "Ambiente 2 (Vigor / Actividad Media)",
            3: "Ambiente 3 (Vigor / Actividad Media-Alta)",
            4: "Ambiente 4 (Vigor / Actividad Alta)"
        }
        color_hex_map = {1: "#d73027", 2: "#fdae61", 3: "#a6d96a", 4: "#1a9850"}

    gdf["nombre_zona"] = gdf["zone_id"].map(labels_map)
    gdf["color_hex"] = gdf["zone_id"].map(color_hex_map)

    gdf_metric = gdf.to_crs(epsg=3857)
    gdf["area_ha"] = (gdf_metric.geometry.area / 10000.0).round(2)

    return gdf


# ---------------------------------------------------------------------------
# 7. GENERADOR DE PUNTOS DE MUESTREO INTELIGENTES A CAMPO
# ---------------------------------------------------------------------------
def sample_interior_points(geom, k=1):
    """Obtiene k puntos interiores espaciados y representativos dentro de una geometría."""
    if geom is None or geom.is_empty:
        return []
    if k <= 1:
        return [geom.representative_point()]

    minx, miny, maxx, maxy = geom.bounds
    nx = int(np.ceil(np.sqrt(k * 25)))
    ny = nx
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)

    # Reducción suave de bordes para evitar puntos en el límite exterior
    span = min(maxx - minx, maxy - miny)
    buf_dist = 0.03 * span if span > 0 else 0
    buf = geom.buffer(-buf_dist) if buf_dist > 0 else geom
    target = buf if (buf is not None and not buf.is_empty) else geom

    candidates = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if target.contains(p):
                candidates.append(p)

    if not candidates:
        return [geom.representative_point()]

    if len(candidates) <= k:
        return candidates

    # Selección voraz por máxima distancia entre puntos
    selected = [candidates[len(candidates) // 2]]
    remaining = [c for c in candidates if c != selected[0]]
    while len(selected) < k and remaining:
        best_cand = max(remaining, key=lambda c: min(c.distance(s) for s in selected))
        selected.append(best_cand)
        remaining.remove(best_cand)

    return selected


def generate_sampling_points(gdf_zones, points_per_zone=2):
    """Genera puntos representativos interiores para cada ambiente de manejo."""
    sampling_records = []
    points_needed = int(points_per_zone)

    for zone_id in sorted(gdf_zones["zone_id"].unique()):
        sub_gdf = gdf_zones[gdf_zones["zone_id"] == zone_id].sort_values("area_ha", ascending=False)
        if sub_gdf.empty:
            continue

        zone_label = sub_gdf["nombre_zona"].iloc[0]
        color_hex = sub_gdf["color_hex"].iloc[0]
        n_polys = len(sub_gdf)
        pts_zone_count = 0

        # Caso 1: Hay suficientes polígonos grandes -> 1 punto representativo por polígono
        if n_polys >= points_needed:
            for _, row in sub_gdf.head(points_needed).iterrows():
                geom = row.geometry
                pts = sample_interior_points(geom, 1)
                for pt in pts:
                    pts_zone_count += 1
                    lat, lon = pt.y, pt.x
                    p_id = f"A{zone_id}_P{pts_zone_count}"
                    sampling_records.append({
                        "id_muestreo": p_id,
                        "zona_id": zone_id,
                        "nombre_zona": zone_label,
                        "color_hex": color_hex,
                        "lat_decimal": round(lat, 6),
                        "lon_decimal": round(lon, 6),
                        "lat_sexagesimal": decimal_to_dms(lat, is_lat=True),
                        "lon_sexagesimal": decimal_to_dms(lon, is_lat=False),
                        "poligono_ha": row["area_ha"],
                        "geometry": pt
                    })
        # Caso 2: Menos polígonos que puntos pedidos -> Repartir cupos proporcionales a la superficie
        else:
            total_zone_area = sub_gdf["area_ha"].sum()
            quotas = []
            for _, row in sub_gdf.iterrows():
                prop = (row["area_ha"] / total_zone_area) if total_zone_area > 0 else (1.0 / n_polys)
                q = max(1, int(round(prop * points_needed)))
                quotas.append(q)

            diff = points_needed - sum(quotas)
            quotas[0] = max(1, quotas[0] + diff)

            for (_, row), q_pts in zip(sub_gdf.iterrows(), quotas):
                geom = row.geometry
                pts = sample_interior_points(geom, q_pts)
                for pt in pts:
                    pts_zone_count += 1
                    lat, lon = pt.y, pt.x
                    p_id = f"A{zone_id}_P{pts_zone_count}"
                    sampling_records.append({
                        "id_muestreo": p_id,
                        "zona_id": zone_id,
                        "nombre_zona": zone_label,
                        "color_hex": color_hex,
                        "lat_decimal": round(lat, 6),
                        "lon_decimal": round(lon, 6),
                        "lat_sexagesimal": decimal_to_dms(lat, is_lat=True),
                        "lon_sexagesimal": decimal_to_dms(lon, is_lat=False),
                        "poligono_ha": row["area_ha"],
                        "geometry": pt
                    })

    if not sampling_records:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(sampling_records, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# 8. CURVA FENOLÓGICA TEMPORAL (SERIE DE TIEMPO MULTITEMPORAL)
# ---------------------------------------------------------------------------
def calculate_temporal_phenology(aoi_ee, classified_img, start_date, end_date, max_cloud_pct, num_zones):
    """Calcula la serie temporal de evolución fenológica del NDVI mes a mes."""
    try:
        intervals = []
        curr = start_date
        total_days = (end_date - start_date).days
        step_days = 30 if total_days > 120 else 15

        while curr < end_date:
            next_d = min(curr + datetime.timedelta(days=step_days), end_date)
            intervals.append((curr.strftime("%Y-%m-%d"), next_d.strftime("%Y-%m-%d")))
            curr = next_d

        if num_zones == 3:
            labels_map = {
                1: "Ambiente 1 (Vigor / Actividad Baja)",
                2: "Ambiente 2 (Vigor / Actividad Media)",
                3: "Ambiente 3 (Vigor / Actividad Alta)"
            }
            color_hex_map = {1: "#d73027", 2: "#fee08b", 3: "#1a9850"}
        else:
            labels_map = {
                1: "Ambiente 1 (Vigor / Actividad Baja)",
                2: "Ambiente 2 (Vigor / Actividad Media)",
                3: "Ambiente 3 (Vigor / Actividad Media-Alta)",
                4: "Ambiente 4 (Vigor / Actividad Alta)"
            }
            color_hex_map = {1: "#d73027", 2: "#fdae61", 3: "#a6d96a", 4: "#1a9850"}

        rows = []
        for d_s, d_e in intervals:
            s2_sub = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi_ee)
                .filterDate(d_s, d_e)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct + 15))
            )
            count = s2_sub.size().getInfo()
            if count > 0:
                s2_clean = s2_sub.map(mask_s2_clouds_scl)
                med_img = s2_clean.median()
                ndvi = med_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
                
                grouped = ndvi.addBands(classified_img).reduceRegion(
                    reducer=ee.Reducer.mean().group(1, "zone"),
                    geometry=aoi_ee,
                    scale=20,
                    maxPixels=1e6
                ).getInfo()

                for g in grouped.get("groups", []):
                    z_id = int(g.get("zone", 1))
                    m_val = g.get("mean")
                    if m_val is not None and z_id in labels_map:
                        rows.append({
                            "Fecha": d_s,
                            "zone_id": z_id,
                            "Ambiente": labels_map[z_id],
                            "NDVI": round(float(m_val), 3),
                            "color_hex": color_hex_map[z_id]
                        })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 9. GENERADORES KML Y GEOJSON (CON POLÍGONOS Y WAYPOINTS)
# ---------------------------------------------------------------------------
def generate_kml_string(gdf_polygons, gdf_points=None, parcel_name="Lote_Ambientado"):
    """Genera KML OGC estructurado con polígonos translúcidos y Waypoints de muestreo."""
    kml_colors = {
        1: "7f1414d7",  # Rojo translúcido
        2: "7f4baefd",  # Naranja
        3: "7f6ad9a6",  # Verde claro
        4: "7f30981a",  # Verde oscuro
    }

    doc = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(doc, "Document")
    name_elem = ET.SubElement(document, "name")
    name_elem.text = f"Ambientacion_{parcel_name}"

    for z_id, kml_col in kml_colors.items():
        style = ET.SubElement(document, "Style", id=f"style_zone_{z_id}")
        line_style = ET.SubElement(style, "LineStyle")
        ET.SubElement(line_style, "color").text = "ff000000"
        ET.SubElement(line_style, "width").text = "1.5"
        poly_style = ET.SubElement(style, "PolyStyle")
        ET.SubElement(poly_style, "color").text = kml_col
        ET.SubElement(poly_style, "fill").text = "1"
        ET.SubElement(poly_style, "outline").text = "1"

    folder_zones = ET.SubElement(document, "Folder")
    ET.SubElement(folder_zones, "name").text = "Ambientes Homogéneos (Polígonos)"

    for zone_id in sorted(gdf_polygons["zone_id"].unique()):
        sub_gdf = gdf_polygons[gdf_polygons["zone_id"] == zone_id]
        zone_label = sub_gdf["nombre_zona"].iloc[0]
        sub_folder = ET.SubElement(folder_zones, "Folder")
        ET.SubElement(sub_folder, "name").text = zone_label

        for idx, row in sub_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            placemark = ET.SubElement(sub_folder, "Placemark")
            ET.SubElement(placemark, "name").text = f"{zone_label} (Polígono {idx})"
            ET.SubElement(placemark, "styleUrl").text = f"#style_zone_{zone_id}"

            ext_data = ET.SubElement(placemark, "ExtendedData")
            d1 = ET.SubElement(ext_data, "Data", name="Ambiente_ID")
            ET.SubElement(d1, "value").text = str(row["zone_id"])
            d2 = ET.SubElement(ext_data, "Data", name="Ambiente_Nombre")
            ET.SubElement(d2, "value").text = str(row["nombre_zona"])
            d3 = ET.SubElement(ext_data, "Data", name="Superficie_ha")
            ET.SubElement(d3, "value").text = str(row["area_ha"])

            single_polys = [geom] if isinstance(geom, Polygon) else geom.geoms
            multi_geom_elem = ET.SubElement(placemark, "MultiGeometry") if len(single_polys) > 1 else placemark

            for poly in single_polys:
                poly_elem = ET.SubElement(multi_geom_elem, "Polygon")
                outer_elem = ET.SubElement(poly_elem, "outerBoundaryIs")
                ring_elem = ET.SubElement(outer_elem, "LinearRing")
                coords_text = " ".join([f"{x},{y},0" for x, y in poly.exterior.coords])
                ET.SubElement(ring_elem, "coordinates").text = coords_text

    if gdf_points is not None and not gdf_points.empty:
        folder_pts = ET.SubElement(document, "Folder")
        ET.SubElement(folder_pts, "name").text = "📍 Puntos de Muestreo a Campo (GPS)"

        for _, pt_row in gdf_points.iterrows():
            p_mark = ET.SubElement(folder_pts, "Placemark")
            ET.SubElement(p_mark, "name").text = f"Muestreo {pt_row['id_muestreo']} ({pt_row['nombre_zona']})"
            desc_text = (
                f"Ambiente: {pt_row['nombre_zona']}\n"
                f"Latitud Decimal: {pt_row['lat_decimal']}\n"
                f"Longitud Decimal: {pt_row['lon_decimal']}\n"
                f"Coordenadas Sexagesimales: {pt_row['lat_sexagesimal']}, {pt_row['lon_sexagesimal']}\n"
                f"Superficie del Paño: {pt_row['poligono_ha']} ha"
            )
            ET.SubElement(p_mark, "description").text = desc_text
            
            point_elem = ET.SubElement(p_mark, "Point")
            ET.SubElement(point_elem, "coordinates").text = f"{pt_row['lon_decimal']},{pt_row['lat_decimal']},0"

    return ET.tostring(doc, encoding="utf-8", method="xml").decode("utf-8")


def add_ee_layer_to_map(folium_map, ee_image_object, vis_params, name, shown=True):
    """Agrega una capa ráster de Google Earth Engine a un mapa Folium."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    tile_layer = folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=shown
    )
    tile_layer.add_to(folium_map)
    return tile_layer


def get_ee_tile_layer(ee_image_object, vis_params, name, shown=True):
    """Crea y retorna un TileLayer de Earth Engine para usar en Folium o SideBySide."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    return folium.raster_layers.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=shown
    )


# ---------------------------------------------------------------------------
# 10. APLICACIÓN PRINCIPAL (STREAMLIT UI)
# ---------------------------------------------------------------------------
def main():
    # Encabezado institucional limpio (el logo permanece en la barra lateral)
    st.markdown('<div class="app-title">INTA AMBIENTE</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Plataforma de Ambientación Multivariada, Fenología Satelital y Muestreo a Campo</div>', unsafe_allow_html=True)
    st.markdown('<span class="app-badge">Proyecto PATCHACO • Grupo RRNN INTA EEA Salta</span>', unsafe_allow_html=True)

    st.markdown("---")

    ee_ok, ee_msg = init_earth_engine()
    if not ee_ok:
        st.error(f"⚠️ Error de Autenticación con Google Earth Engine: {ee_msg}")
        return

    # Inicializar session_state
    if "current_aoi" not in st.session_state:
        st.session_state["current_aoi"] = None
    if "aoi_name" not in st.session_state:
        st.session_state["aoi_name"] = "Parcela_Agricola"
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "last_aoi_source" not in st.session_state:
        st.session_state["last_aoi_source"] = None
    if "geo_pin" not in st.session_state:
        st.session_state["geo_pin"] = None

    # Sidebar: Logo y Configuración
    with st.sidebar:
        if os.path.exists("logo_inta.png"):
            st.image("logo_inta.png", width=140)
        st.markdown("### **INTA AMBIENTE**")
        st.caption("Plataforma Satelital de Manejo por Ambientes")

    st.sidebar.header("⚙️ Parámetros de Análisis")

    # 1. Definición del AOI
    st.sidebar.subheader("1. Área de Interés (AOI)")
    aoi_source = st.sidebar.radio(
        "Método de entrada de la parcela:",
        ["✏️ Dibujar en el Mapa", "📁 Subir KML / GeoJSON"],
        index=0
    )

    if st.session_state["last_aoi_source"] is not None and aoi_source != st.session_state["last_aoi_source"]:
        st.session_state["current_aoi"] = None
        st.session_state["results"] = None
        st.session_state["aoi_name"] = "Parcela_Agricola"
        st.session_state["last_aoi_source"] = aoi_source
        st.rerun()

    st.session_state["last_aoi_source"] = aoi_source

    if aoi_source == "📁 Subir KML / GeoJSON":
        uploaded_file = st.sidebar.file_uploader(
            "Cargar archivo de lote (.kml, .geojson, .json):",
            type=["kml", "geojson", "json"]
        )
        st.sidebar.caption("💡 Límite operativo: hasta **10.000 ha** por parcela.")
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            fname = uploaded_file.name.lower()
            st.session_state["aoi_name"] = uploaded_file.name.rsplit(".", 1)[0]
            if fname.endswith(".kml"):
                geom, err = parse_kml_to_shapely(file_bytes)
            else:
                geom, err = parse_geojson_to_shapely(file_bytes)

            if err:
                st.sidebar.error(err)
            else:
                st.session_state["current_aoi"] = geom
                st.sidebar.success(f"Lote cargado: {st.session_state['aoi_name']}")

    # --- Herramienta de Geoposicionamiento ---
    with st.sidebar.expander("📍 Geoposicionar / Ir a Coordenadas", expanded=False):
        st.caption("Ingresa coordenadas Decimales o Sexagesimales (DMS) para centrar el mapa antes de dibujar:")
        coord_tab1, coord_tab2 = st.tabs(["Decimal", "Sexagesimal (DMS)"])
        
        with coord_tab1:
            dec_input = st.text_input("Latitud, Longitud:", placeholder="Ej: -24.9015, -65.4820", key="geo_dec_input")
            if st.button("📍 Centrar por Decimal", use_container_width=True):
                lat, lon = parse_free_coordinate_string(dec_input)
                if lat is not None and lon is not None:
                    st.session_state["geo_pin"] = [lat, lon]
                    st.success(f"Ubicación fijada: {lat:.5f}, {lon:.5f}")
                    st.rerun()
                else:
                    st.error("Formato no válido. Usa ej: -24.9015, -65.4820")

        with coord_tab2:
            st.caption("Ejemplo: 24° 54' 05\" S, 65° 28' 55\" O")
            dms_input = st.text_input("Coordenadas DMS:", placeholder="24° 54' 05\" S, 65° 28' 55\" O", key="geo_dms_input")
            if st.button("📍 Centrar por Sexagesimal", use_container_width=True):
                lat, lon = parse_free_coordinate_string(dms_input)
                if lat is not None and lon is not None:
                    st.session_state["geo_pin"] = [lat, lon]
                    st.success(f"Ubicación fijada: {lat:.5f}, {lon:.5f}")
                    st.rerun()
                else:
                    st.error("Formato no reconocido. Prueba ej: 24° 54' 05\" S, 65° 28' 55\" O")

    # 2. Sentinel-2
    st.sidebar.subheader("2. Filtro Sentinel-2")
    col_d1, col_d2 = st.sidebar.columns(2)
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=90)
    
    start_date = col_d1.date_input("Fecha Inicio:", value=default_start)
    end_date = col_d2.date_input("Fecha Fin:", value=today)

    max_cloud_pct = st.sidebar.slider(
        "Cobertura máxima de nubes (%):",
        min_value=1,
        max_value=60,
        value=15,
        step=5
    )

    # 3. Estrategia de Ambientación y Selección de Índices
    st.sidebar.subheader("3. Estrategia Agronómica")
    
    ambientacion_mode = st.sidebar.selectbox(
        "Criterio de Segmentación:",
        [
            "🌟 Multivariada Completa (NDVI + NDRE + NDWI)",
            "🌿 Monovariable: NDVI (Biomasa & Cobertura)",
            "🌾 Monovariable: NDRE (Red Edge - Clorofila / Sin Saturación)",
            "💧 Monovariable: NDWI (SWIR - Contenido Hídrico & Suelo)",
            "🎯 Monovariable: ICA (Índice Compuesto Integrado 40/35/25)"
        ],
        index=0
    )

    num_zones = st.sidebar.radio(
        "Cantidad de Zonas de Manejo:",
        options=[3, 4],
        index=1,
        format_func=lambda x: f"{x} Zonas Homogéneas"
    )

    if "Multivariada" in ambientacion_mode:
        clustering_algo = "Weka K-Means Multivariado (GEE)"
    else:
        clustering_algo = st.sidebar.selectbox(
            "Algoritmo de Segmentación:",
            ["Cuantiles Agronómicos (Percentiles)", "Weka K-Means (Clustering GEE)"]
        )

    smooth_filter = st.sidebar.checkbox(
        "Aplicar Suavizado Espacial (Focal Mode 10m)",
        value=True,
        help="Elimina ruido de píxeles aislados para crear paños limpios adaptados a prescripciones de siembra y fertilización variable."
    )

    # Verificación de área del lote para tope dinámico y validación de seguridad
    ha_chk = 0.0
    if st.session_state["current_aoi"] is not None:
        try:
            gdf_chk = gpd.GeoDataFrame(geometry=[st.session_state["current_aoi"]], crs="EPSG:4326").to_crs(epsg=3857)
            ha_chk = round(gdf_chk.geometry.area.iloc[0] / 10000.0, 2)
        except Exception:
            pass

    # Tope dinámico agronómico según la superficie del polígono:
    if ha_chk <= 0:
        max_pts_allowed = 10
        cap_note = "Hasta 10 puntos por ambiente (se adaptará al cargar o dibujar el lote)."
    elif ha_chk < 20:
        max_pts_allowed = 4
        cap_note = f"Tope de {max_pts_allowed} puntos/ambiente en lotes pequeños ({ha_chk:.1f} ha) para evitar aglomeración."
    elif ha_chk < 100:
        max_pts_allowed = 6
        cap_note = f"Tope de {max_pts_allowed} puntos/ambiente recomendado para {ha_chk:.1f} ha."
    elif ha_chk < 500:
        max_pts_allowed = 10
        cap_note = f"Tope de {max_pts_allowed} puntos/ambiente para {ha_chk:.1f} ha."
    else:
        max_pts_allowed = 15
        cap_note = f"Tope de {max_pts_allowed} puntos/ambiente para grandes extensiones ({ha_chk:,.1f} ha)."

    points_per_zone = st.sidebar.number_input(
        "Puntos de Muestreo por Ambiente (GPS):",
        min_value=1,
        max_value=max_pts_allowed,
        value=min(2, max_pts_allowed),
        step=1,
        help="Elige con las flechas o tipea directamente el número de sitios a muestrear por cada ambiente delimitado."
    )
    total_pts_estimados = points_per_zone * num_zones
    st.sidebar.caption(f"📍 **Total:** {total_pts_estimados} puntos en el lote ({points_per_zone} por cada uno de los {num_zones} ambientes).")
    st.sidebar.caption(f"💡 *{cap_note}*")

    sidebar_disabled = (ha_chk > MAX_SUPERFICIE_HA)

    run_btn = st.sidebar.button(
        "🚀 Procesar Ambientación",
        use_container_width=True,
        disabled=sidebar_disabled
    ) or (st.session_state.pop("trigger_process", False) and not sidebar_disabled)

    if st.session_state["current_aoi"] is not None:
        if st.sidebar.button("🔄 Cambiar Lote / Nuevo Dibujo", use_container_width=True):
            st.session_state["current_aoi"] = None
            st.session_state["results"] = None
            st.rerun()

    # Créditos en Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Diseño y Metodología:**  
    👨‍💻 **Lic. Inf. Hernan Elena**  
    *Grupo de Recursos Naturales (RRNN)*  
    **INTA EEA Salta** • *Proyecto PATCHACO*
    """)

    # --- PANTALLA PRINCIPAL ---
    
    # CASO A: Aún no tenemos polígono
    if st.session_state["current_aoi"] is None and aoi_source == "✏️ Dibujar en el Mapa":
        st.markdown("### 🗺️ Paso 1: Dibuja tu parcela en el mapa interactivo")
        st.caption("Navega por el mapa o usa la herramienta de geoposicionamiento en la barra lateral. Utiliza el icono de polígono o rectángulo (arriba a la izquierda) para trazar los límites del lote (límite operativo máximo: **10.000 ha**).")

        map_center = st.session_state["geo_pin"] if st.session_state["geo_pin"] else [-24.90, -65.48]
        map_zoom = 15 if st.session_state["geo_pin"] else 13
        
        # Google Satellite SIEMPRE por defecto como base principal con nombre limpio
        m_draw = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles=None
        )
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Google Satélite (Alta Resolución)",
            overlay=False,
            control=True
        ).add_to(m_draw)

        # Argenmap agregada como capa base secundaria (show=False)
        folium.TileLayer(
            tiles="https://wms.ign.gob.ar/geoserver/gwc/service/wmts/rest/capabaseargenmap/normal/EPSG:3857/EPSG:3857:{z}/{y}/{x}?format=image/png",
            attr="Instituto Geográfico Nacional (IGN Argentina)",
            name="Argenmap (IGN Argentina)",
            overlay=False,
            control=True,
            show=False
        ).add_to(m_draw)

        if st.session_state["geo_pin"]:
            folium.Marker(
                st.session_state["geo_pin"],
                popup=f"📍 Coordenadas: {st.session_state['geo_pin'][0]:.5f}, {st.session_state['geo_pin'][1]:.5f}",
                tooltip="Punto Geoposicionado de Referencia",
                icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
            ).add_to(m_draw)

        draw = Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "polygon": True,
                "rectangle": True
            }
        )
        draw.add_to(m_draw)
        folium.LayerControl(position="topright", collapsed=False).add_to(m_draw)
        Fullscreen().add_to(m_draw)

        map_output = st_folium(m_draw, use_container_width=True, height=530, key="draw_map")
        
        if map_output and map_output.get("last_active_drawing"):
            geom_dict = map_output["last_active_drawing"].get("geometry")
            if geom_dict:
                geom, err = parse_geojson_to_shapely(geom_dict)
                if not err:
                    st.session_state["current_aoi"] = geom
                    st.session_state["aoi_name"] = "Lote_Dibujado"
                    st.rerun()

    # CASO B: Ya tenemos AOI delimitado
    if st.session_state["current_aoi"] is not None:
        shapely_aoi = st.session_state["current_aoi"]
        ee_aoi = shapely_to_ee_geometry(shapely_aoi)
        centroid = shapely_aoi.centroid
        center_lat, center_lon = centroid.y, centroid.x

        gdf_temp = gpd.GeoDataFrame(geometry=[shapely_aoi], crs="EPSG:4326").to_crs(epsg=3857)
        lote_ha = round(gdf_temp.geometry.area.iloc[0] / 10000.0, 2)
        excede_limite = lote_ha > MAX_SUPERFICIE_HA

        # Banner de Métricas Responsivas (diseño que no se desborda en pantalla)
        strategy_short = ambientacion_mode.split(":")[0].replace("🌟", "").replace("🌿", "").replace("🌾", "").replace("💧", "").replace("🎯", "").strip()
        metric_sup_color = "color: #e63946; font-weight: bold;" if excede_limite else ""
        metric_sup_sub = "⚠️ Excede 10.000 ha" if excede_limite else "Área de Parcela"

        st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-pill">
                <div class="metric-pill-label">Superficie Total</div>
                <div class="metric-pill-value" style="{metric_sup_color}">{lote_ha} ha</div>
                <div class="metric-pill-sub">{metric_sup_sub}</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Período Temporal</div>
                <div class="metric-pill-value">{(end_date - start_date).days} días</div>
                <div class="metric-pill-sub">{start_date.strftime('%d/%m/%y')} al {end_date.strftime('%d/%m/%y')}</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Filtro de Nubes</div>
                <div class="metric-pill-value">&lt; {max_cloud_pct} %</div>
                <div class="metric-pill-sub">Máscara SCL Activa</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-label">Estrategia Aplicada</div>
                <div class="metric-pill-value">{strategy_short}</div>
                <div class="metric-pill-sub">{num_zones} Ambientes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if excede_limite:
            st.error(
                f"⛔ **Límite de superficie superado ({lote_ha:,.2f} ha):**\n\n"
                f"El tamaño máximo admitido por análisis es de **{int(MAX_SUPERFICIE_HA):,} hectáreas**. "
                f"Este control de seguridad previene la saturación de cuota y tiempos de espera excesivos en Google Earth Engine, "
                f"garantizando que la plataforma se mantenga ágil y estable para todos los usuarios.\n\n"
                f"👉 **Recomendación:** Por favor, acota el dibujo en el mapa o divide tu archivo KML/GeoJSON en potreros o sectores menores a 10.000 ha."
            )

        # Si aún no se procesó, mostrar vista previa del lote y botón de inicio
        if st.session_state["results"] is None:
            st.markdown("### 📍 Vista Previa del Lote Delimitado")
            st.info(f"Parcela: **{st.session_state['aoi_name']}** — Superficie: **{lote_ha} hectáreas**. Revisa los límites y presiona el botón para procesar la ambientación.")

            # Google Satellite SIEMPRE por defecto con nombre limpio
            m_prev = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles=None
            )
            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google Satellite",
                name="Google Satélite (Alta Resolución)",
                overlay=False,
                control=True
            ).add_to(m_prev)
            # Argenmap secundaria
            folium.TileLayer(
                tiles="https://wms.ign.gob.ar/geoserver/gwc/service/wmts/rest/capabaseargenmap/normal/EPSG:3857/EPSG:3857:{z}/{y}/{x}?format=image/png",
                attr="Instituto Geográfico Nacional (IGN Argentina)",
                name="Argenmap (IGN Argentina)",
                overlay=False,
                control=True,
                show=False
            ).add_to(m_prev)

            folium.GeoJson(
                mapping(shapely_aoi),
                name="Perímetro de Parcela",
                style_function=lambda x: {
                    "fillColor": "#00f0ff",
                    "fillOpacity": 0.2,
                    "color": "#00f0ff",
                    "weight": 3.5
                }
            ).add_to(m_prev)
            minx, miny, maxx, maxy = shapely_aoi.bounds
            m_prev.fit_bounds([[miny, minx], [maxy, maxx]])
            folium.LayerControl(position="topright", collapsed=False).add_to(m_prev)
            Fullscreen().add_to(m_prev)
            st_folium(m_prev, use_container_width=True, height=460, key="preview_map", returned_objects=[])

            col_btn1, col_btn2 = st.columns([3, 1])
            if excede_limite:
                col_btn1.button("🚫 Procesamiento Deshabilitado (Supera 10.000 ha)", disabled=True, use_container_width=True)
            else:
                if col_btn1.button("🚀 Iniciar Procesamiento de Ambientación", type="primary", use_container_width=True):
                    st.session_state["trigger_process"] = True
                    st.rerun()

            if col_btn2.button("🗑️ Descartar Lote / Volver", use_container_width=True):
                st.session_state["current_aoi"] = None
                st.session_state["results"] = None
                st.rerun()

        # Procesamiento
        if run_btn and not excede_limite:
            with st.spinner("🛰️ Consultando Sentinel-2 Harmonized SR y procesando ambientación multiespectral..."):
                try:
                    composite, n_images = get_sentinel2_composite(ee_aoi, start_date, end_date, max_cloud_pct)
                    if composite is None or n_images == 0:
                        st.error(f"No se encontraron escenas de Sentinel-2 para el rango {start_date} al {end_date} con nubes < {max_cloud_pct}%. Amplía el rango de fechas.")
                        return

                    img_indices = calculate_agronomic_indices(composite)

                    if "Multivariada" in ambientacion_mode:
                        classified, metrics = classify_zones_multivariate_kmeans(
                            img_indices, ee_aoi, num_zones=num_zones, smooth=smooth_filter
                        )
                        mode_tag = "Multivariada (NDVI + NDRE + NDWI)"
                    else:
                        if "NDRE" in ambientacion_mode:
                            t_band = "NDRE"
                        elif "NDWI" in ambientacion_mode:
                            t_band = "NDWI"
                        elif "ICA" in ambientacion_mode:
                            t_band = "ICA"
                        else:
                            t_band = "NDVI"

                        classified, metrics = classify_zones_quantiles(
                            img_indices, ee_aoi, target_band=t_band, num_zones=num_zones, smooth=smooth_filter
                        )
                        mode_tag = f"Monovariable ({t_band})"

                    gdf_zones = vectorize_zones_to_gdf(classified, ee_aoi, num_zones=num_zones)
                    gdf_points = generate_sampling_points(gdf_zones, points_per_zone=points_per_zone)

                    with st.spinner("📈 Calculando serie temporal de evolución fenológica por ambiente..."):
                        df_phenology = calculate_temporal_phenology(
                            ee_aoi, classified, start_date, end_date, max_cloud_pct, num_zones
                        )

                    st.session_state["results"] = {
                        "n_images": n_images,
                        "composite": composite,
                        "img_indices": img_indices,
                        "classified": classified,
                        "metrics": metrics,
                        "gdf_zones": gdf_zones,
                        "gdf_points": gdf_points,
                        "df_phenology": df_phenology,
                        "num_zones": num_zones,
                        "mode_tag": mode_tag,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "lote_ha": lote_ha
                    }
                    st.success("✅ ¡Ambientación y serie temporal generadas exitosamente!")
                except Exception as ex:
                    st.error(f"Error durante el procesamiento: {str(ex)}")
                    import traceback
                    st.code(traceback.format_exc())

        # Si tenemos resultados procesados, mostrarlos en pantalla
        if st.session_state["results"] is not None:
            res = st.session_state["results"]
            composite = res["composite"]
            img_indices = res["img_indices"]
            classified = res["classified"]
            gdf_zones = res["gdf_zones"]
            gdf_points = res.get("gdf_points", gpd.GeoDataFrame())
            df_phenology = res.get("df_phenology", pd.DataFrame())
            metrics = res["metrics"]
            res_num_zones = res["num_zones"]
            res_mode = res.get("mode_tag", "Ambientación")

            st.markdown("---")
            
            tab_mapa, tab_curva, tab_muestreo, tab_estadisticas, tab_descargas, tab_manual = st.tabs([
                "🗺️ Visualizador Espacial",
                "📈 Curva Fenológica Temporal",
                "📍 Puntos de Muestreo (GPS)",
                "📊 Diagnóstico y Estadísticas",
                "💾 Exportación y Descarga",
                "📖 Manual & Metodología"
            ])

            # ---------------------------------------------------------------
            # PESTAÑA 1: VISUALIZADOR ESPACIAL (ESTÁNDAR O SWIPE DESLIZANTE)
            # ---------------------------------------------------------------
            with tab_mapa:
                st.markdown(f"#### 🌾 Mapa de Ambientes Homogéneos — `{res_mode}`")

                # Selector de Modo de Visualización: Multicapa o SWIPE Deslizante
                col_view_mode, col_swipe_ctrl = st.columns([1, 2])
                with col_view_mode:
                    map_mode = st.radio(
                        "Modo de Visualización:",
                        ["🎛️ Multicapa Estándar", "↔️ Cortina Deslizante (SWIPE)"],
                        horizontal=True
                    )

                # Leyenda gráfica visual
                if res_num_zones == 3:
                    st.markdown("""
                    <div class="legend-box">
                        <span class="legend-item"><span class="legend-color" style="background-color: #d73027;"></span>Ambiente 1 (Vigor / Actividad Baja)</span>
                        <span class="legend-item"><span class="legend-color" style="background-color: #fee08b;"></span>Ambiente 2 (Vigor / Actividad Media)</span>
                        <span class="legend-item"><span class="legend-color" style="background-color: #1a9850;"></span>Ambiente 3 (Vigor / Actividad Alta)</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="legend-box">
                        <span class="legend-item"><span class="legend-color" style="background-color: #d73027;"></span>Ambiente 1 (Vigor / Actividad Baja)</span>
                        <span class="legend-item"><span class="legend-color" style="background-color: #fdae61;"></span>Ambiente 2 (Vigor / Actividad Media)</span>
                        <span class="legend-item"><span class="legend-color" style="background-color: #a6d96a;"></span>Ambiente 3 (Vigor / Actividad Media-Alta)</span>
                        <span class="legend-item"><span class="legend-color" style="background-color: #1a9850;"></span>Ambiente 4 (Vigor / Actividad Alta)</span>
                    </div>
                    """, unsafe_allow_html=True)

                if res_num_zones == 3:
                    palette_zones = ["#d73027", "#fee08b", "#1a9850"]
                else:
                    palette_zones = ["#d73027", "#fdae61", "#a6d96a", "#1a9850"]
                vis_zones = {"min": 1, "max": res_num_zones, "palette": palette_zones, "opacity": 0.85}

                vis_ndvi = {"bands": ["NDVI"], "min": 0.15, "max": 0.85, "palette": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"]}
                vis_ndre = {"bands": ["NDRE"], "min": 0.10, "max": 0.65, "palette": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]}
                vis_ndwi = {"bands": ["NDWI"], "min": -0.15, "max": 0.45, "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]}
                vis_rgb = {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.28, "gamma": 1.3}
                vis_cir = {"bands": ["B8", "B4", "B3"], "min": 0.0, "max": 0.38, "gamma": 1.3}

                # --- MODO 1: CORTINA DESLIZANTE (SWIPE) ---
                if "SWIPE" in map_mode:
                    with col_swipe_ctrl:
                        col_sw1, col_sw2 = st.columns(2)
                        swipe_left_name = col_sw1.selectbox(
                            "Capa Izquierda (Base):",
                            ["Google Satélite", "Color Natural (RGB)", "Falso Color CIR", "NDVI Continuo"],
                            index=0
                        )
                        swipe_right_name = col_sw2.selectbox(
                            "Capa Derecha (Comparar):",
                            ["Ambientación (Zonas de Manejo)", "NDVI Continuo", "NDRE (Red Edge)", "NDWI (Hídrico SWIR)"],
                            index=0
                        )

                    st.caption("↔️ **Mueve la barra vertical divisoria en el mapa** con el cursor para contrastar la capa izquierda y derecha en tiempo real.")

                    m_swipe = folium.Map(
                        location=[res["center_lat"], res["center_lon"]],
                        zoom_start=15,
                        tiles=None
                    )

                    # Construir capa izquierda
                    if swipe_left_name == "Google Satélite":
                        layer_left = folium.TileLayer(
                            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                            attr="Google Satellite",
                            name="Google Satélite"
                        )
                    elif swipe_left_name == "Color Natural (RGB)":
                        layer_left = get_ee_tile_layer(composite, vis_rgb, "Color Natural RGB")
                    elif swipe_left_name == "Falso Color CIR":
                        layer_left = get_ee_tile_layer(composite, vis_cir, "Falso Color CIR")
                    else:
                        layer_left = get_ee_tile_layer(img_indices.select("NDVI"), vis_ndvi, "NDVI Continuo")

                    # Construir capa derecha
                    if swipe_right_name == "Ambientación (Zonas de Manejo)":
                        layer_right = get_ee_tile_layer(classified, vis_zones, "Ambientación")
                    elif swipe_right_name == "NDVI Continuo":
                        layer_right = get_ee_tile_layer(img_indices.select("NDVI"), vis_ndvi, "NDVI Continuo")
                    elif swipe_right_name == "NDRE (Red Edge)":
                        layer_right = get_ee_tile_layer(img_indices.select("NDRE"), vis_ndre, "NDRE")
                    else:
                        layer_right = get_ee_tile_layer(img_indices.select("NDWI"), vis_ndwi, "NDWI")

                    layer_left.add_to(m_swipe)
                    layer_right.add_to(m_swipe)

                    # Plugin Side by Side (Swipe)
                    sbs = SideBySideLayers(layer_left=layer_left, layer_right=layer_right)
                    sbs.add_to(m_swipe)

                    # Contorno de Parcela
                    folium.GeoJson(
                        mapping(shapely_aoi),
                        name="Perímetro de Parcela (AOI)",
                        style_function=lambda x: {"fillColor": "transparent", "color": "#00f0ff", "weight": 3.5, "dashArray": "6, 6"}
                    ).add_to(m_swipe)

                    Fullscreen().add_to(m_swipe)
                    st_folium(m_swipe, use_container_width=True, height=600, key="swipe_result_map", returned_objects=[])

                # --- MODO 2: MULTICAPA ESTÁNDAR ---
                else:
                    st.caption("Usa el menú de capas (arriba a la derecha) para alternar entre Argenmap, Google Satélite, Ambientación, Puntos de Muestreo e Índices.")

                    m = folium.Map(
                        location=[res["center_lat"], res["center_lon"]],
                        zoom_start=15,
                        tiles=None
                    )
                    folium.TileLayer(
                        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                        attr="Google Satellite",
                        name="Google Satélite (Alta Resolución)",
                        overlay=False,
                        control=True
                    ).add_to(m)

                    folium.TileLayer(
                        tiles="https://wms.ign.gob.ar/geoserver/gwc/service/wmts/rest/capabaseargenmap/normal/EPSG:3857/EPSG:3857:{z}/{y}/{x}?format=image/png",
                        attr="Instituto Geográfico Nacional (IGN Argentina)",
                        name="Argenmap (IGN Argentina)",
                        overlay=False,
                        control=True,
                        show=False
                    ).add_to(m)

                    add_ee_layer_to_map(m, classified, vis_zones, f"🌾 Ambientación ({res_num_zones} Zonas)", shown=True)
                    add_ee_layer_to_map(m, img_indices.select("NDVI"), vis_ndvi, "🌿 Índice NDVI (Biomasa)", shown=False)
                    add_ee_layer_to_map(m, img_indices.select("NDRE"), vis_ndre, "🌾 Índice NDRE (Clorofila Red Edge)", shown=False)
                    add_ee_layer_to_map(m, img_indices.select("NDWI"), vis_ndwi, "💧 Índice NDWI (Humedad SWIR)", shown=False)
                    add_ee_layer_to_map(m, composite, vis_rgb, "🌈 Sentinel-2: Color Natural (RGB)", shown=False)
                    add_ee_layer_to_map(m, composite, vis_cir, "🔴 Sentinel-2: Falso Color CIR", shown=False)

                    folium.GeoJson(
                        mapping(shapely_aoi),
                        name="Perímetro de Parcela (AOI)",
                        style_function=lambda x: {"fillColor": "transparent", "color": "#00f0ff", "weight": 3.5, "dashArray": "6, 6"}
                    ).add_to(m)

                    if not gdf_points.empty:
                        pts_group = folium.FeatureGroup(name="📍 Puntos de Muestreo a Campo", show=True)
                        for _, pt_row in gdf_points.iterrows():
                            folium.Marker(
                                [pt_row["lat_decimal"], pt_row["lon_decimal"]],
                                tooltip=f"Punto {pt_row['id_muestreo']} ({pt_row['nombre_zona']})",
                                popup=(
                                    f"<b>Muestreo {pt_row['id_muestreo']}</b><br>"
                                    f"Ambiente: {pt_row['nombre_zona']}<br>"
                                    f"Decimal: {pt_row['lat_decimal']}, {pt_row['lon_decimal']}<br>"
                                    f"Sexagesimal: {pt_row['lat_sexagesimal']}, {pt_row['lon_sexagesimal']}"
                                ),
                                icon=folium.Icon(color="darkblue", icon="flag", prefix="fa")
                            ).add_to(pts_group)
                        pts_group.add_to(m)

                    folium.LayerControl(position="topright", collapsed=False).add_to(m)
                    Fullscreen().add_to(m)
                    st_folium(m, use_container_width=True, height=600, key="main_result_map", returned_objects=[])

            # ---------------------------------------------------------------
            # PESTAÑA 2: CURVA FENOLÓGICA TEMPORAL (ESTILO FONTAGRO)
            # ---------------------------------------------------------------
            with tab_curva:
                st.markdown("#### 📈 Evolución Fenológica Temporal por Ambiente")
                st.caption("Esta curva muestra la dinámica y estacionalidad del vigor vegetativo (NDVI) a lo largo del tiempo para cada ambiente clasificado:")

                if not df_phenology.empty:
                    fig_pheno = px.line(
                        df_phenology,
                        x="Fecha",
                        y="NDVI",
                        color="Ambiente",
                        color_discrete_map={row["Ambiente"]: row["color_hex"] for _, row in df_phenology.iterrows()},
                        markers=True,
                        title=f"Curva Fenológica Multitemporal (Sentinel-2 SR) — {st.session_state['aoi_name']}"
                    )
                    fig_pheno.update_traces(line=dict(width=3), marker=dict(size=7))
                    fig_pheno.update_layout(
                        xaxis_title="Fecha de Pasada Satelital",
                        yaxis_title="Índice de Vigor (NDVI)",
                        hovermode="x unified",
                        margin=dict(t=50, b=30, l=10, r=10)
                    )
                    st.plotly_chart(fig_pheno, use_container_width=True)

                    st.info("""
                    **Interpretación de la Curva Fenológica:**
                    - **Separación de curvas:** Cuando las líneas de cada ambiente se distancian marcadamente, señala el momento en el que el estrés hídrico o la limitante ambiental comenzó a manifestarse.
                    - **Pico de verdor (Peak NDVI):** Coincide con la máxima actividad biológica y densidad de cobertura foliar.
                    - **Tasas de senescencia:** Una caída rápida en el Ambiente 1 suele indicar pérdida anticipada de verdor o menor retención de humedad en el perfil.
                    """)
                else:
                    st.info("No se pudieron recopilar suficientes escenas limpias en subperíodos para graficar la serie temporal. Intenta ampliar el rango de fechas.")

            # ---------------------------------------------------------------
            # PESTAÑA 3: PUNTOS DE MUESTREO A CAMPO (WAYPOINTS GPS)
            # ---------------------------------------------------------------
            with tab_muestreo:
                st.markdown("#### 📍 Puntos de Muestreo Representativos para GPS y Terreno")
                st.write("Sitios representativos ubicados matemáticamente en el interior de cada ambiente para guiar muestreos de suelo, recuento de plantas o cortes de pasturas a campo:")

                if not gdf_points.empty:
                    table_pts = gdf_points[[
                        "id_muestreo", "nombre_zona", "lat_decimal", "lon_decimal",
                        "lat_sexagesimal", "lon_sexagesimal", "poligono_ha"
                    ]].copy()
                    table_pts.columns = [
                        "ID Muestreo", "Ambiente Homogéneo", "Latitud (Dec)", "Longitud (Dec)",
                        "Latitud (DMS / Sexagesimal)", "Longitud (DMS / Sexagesimal)", "Superficie Paño (ha)"
                    ]
                    st.dataframe(table_pts, use_container_width=True, hide_index=True)

                    csv_pts = table_pts.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Puntos de Muestreo en CSV (para GPS Garmin/Excel)",
                        data=csv_pts,
                        file_name=f"Puntos_Muestreo_{st.session_state['aoi_name']}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No se generaron puntos de muestreo.")

            # ---------------------------------------------------------------
            # PESTAÑA 4: DIAGNÓSTICO Y ESTADÍSTICAS DE AMBIENTES
            # ---------------------------------------------------------------
            with tab_estadisticas:
                st.markdown(f"#### 📊 Caracterización Cuantitativa — `{res_mode}`")

                summary_df = (
                    gdf_zones.groupby(["zone_id", "nombre_zona", "color_hex"])["area_ha"]
                    .sum()
                    .reset_index()
                )
                total_area = summary_df["area_ha"].sum()
                summary_df["porcentaje"] = ((summary_df["area_ha"] / total_area) * 100).round(1)

                col_g1, col_g2 = st.columns([1, 1])

                with col_g1:
                    fig_pie = px.pie(
                        summary_df,
                        values="area_ha",
                        names="nombre_zona",
                        color="nombre_zona",
                        color_discrete_map={row["nombre_zona"]: row["color_hex"] for _, row in summary_df.iterrows()},
                        hole=0.45,
                        title="Proporción de Superficie por Ambiente"
                    )
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_g2:
                    fig_bar = px.bar(
                        summary_df,
                        x="nombre_zona",
                        y="area_ha",
                        color="nombre_zona",
                        color_discrete_map={row["nombre_zona"]: row["color_hex"] for _, row in summary_df.iterrows()},
                        labels={"nombre_zona": "Ambiente", "area_ha": "Superficie (ha)"},
                        title="Superficie Total por Ambiente"
                    )
                    fig_bar.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("##### 📋 Resumen de Parcelamiento:")
                display_table = summary_df[["nombre_zona", "area_ha", "porcentaje"]].copy()
                display_table.columns = ["Ambiente", "Superficie (ha)", "Participación del Lote (%)"]
                st.dataframe(display_table, use_container_width=True, hide_index=True)

                if "Multivariada" in res_mode and isinstance(metrics, dict) and any("Ambiente" in k for k in metrics):
                    st.markdown("##### 🔬 Perfil Espectral y Diagnóstico por Ambiente:")
                    df_profiles = pd.DataFrame.from_dict(metrics, orient="index")
                    df_profiles.index.name = "Ambiente"
                    st.dataframe(df_profiles, use_container_width=True)
                else:
                    st.markdown("##### 🔬 Umbrales del Algoritmo:")
                    for k, v in metrics.items():
                        st.write(f"- Umbral `{k}`: **{v}**")

            # ---------------------------------------------------------------
            # PESTAÑA 5: DESCARGAS DE ARCHIVOS VECTORIALES
            # ---------------------------------------------------------------
            with tab_descargas:
                st.markdown("#### 💾 Exportación para GIS, GPS y Trabajo a Campo")
                st.write("Descarga los polígonos vectorizados de la ambientación junto a los **Puntos de Muestreo (Waypoints)** para analizarlos en software GIS, Google Earth Pro o navegadores GPS para recorridas de terreno:")

                col_down1, col_down2 = st.columns(2)

                kml_str = generate_kml_string(gdf_zones, gdf_points=gdf_points, parcel_name=st.session_state["aoi_name"])
                col_down1.download_button(
                    label="📥 Descargar KML Completo (.kml) [Polígonos + Waypoints GPS]",
                    data=kml_str,
                    file_name=f"Ambientacion_{st.session_state['aoi_name']}_{res_num_zones}zonas_con_muestreos.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    help="Incluye polígonos coloreados por zona y chinchetas de puntos de muestreo listas para Google Earth Pro y GPS."
                )

                geojson_str = gdf_zones.to_json(indent=2)
                col_down2.download_button(
                    label="📥 Descargar Polígonos en GeoJSON (.geojson)",
                    data=geojson_str,
                    file_name=f"Ambientacion_{st.session_state['aoi_name']}_{res_num_zones}zonas.geojson",
                    mime="application/geo+json",
                    help="Formato abierto estándar para QGIS, ArcGIS o desarrollo de scripts."
                )

            # ---------------------------------------------------------------
            # PESTAÑA 6: MANUAL DE USUARIO Y METODOLOGÍA CIENTÍFICA
            # ---------------------------------------------------------------
            with tab_manual:
                st.markdown("### 📖 Manual de Usuario y Metodología Científica")
                st.markdown("""
                #### 1. Marco Metodológico y Fuentes Satelitales
                **INTA AMBIENTE** utiliza la constelación de satélites **Sentinel-2A y Sentinel-2B** de la Agencia Espacial Europea (ESA), a través de la colección **`COPERNICUS/S2_SR_HARMONIZED`** en Google Earth Engine:
                - **Resolución Espacial:** 10 metros nativos en el visible e Infrarrojo Cercano (NIR); 20 metros en Red Edge e Infrarrojo de Onda Corta (SWIR), remuestreados internamente a 10 metros.
                - **Nivel de Procesamiento:** Nivel 2A (Reflectancia de Superficie en el fondo de la atmósfera - BOA), corregido de dispersión y absorción de aerosoles atmosféricos.
                - **Máscara de Nubes Píxel a Píxel (SCL):** Utiliza la banda *Scene Classification Layer* para descartar automáticamente sombras de nubes (clase 3), nubes de media y alta probabilidad (clases 8 y 9), cirros finos (clase 10) y nieve (clase 11).
                - **Reducción Temporal por Mediana:** A diferencia del promedio aritmético simple, la **mediana estadística** evalúa la serie temporal de cada píxel y toma el valor central. Esto garantiza inmunidad total a ruidos de sensor, destellos o nubosidades residuales.
                - **Límite Operativo de Área (10.000 ha):** Para garantizar respuestas en pocos segundos y prevenir la saturación de cuotas y memoria en Google Earth Engine, la plataforma establece un umbral máximo de 10.000 hectáreas (100 km²) por lote analizado, abarcando con holgura parcelas agrícolas, estancias y potreros ganaderos extensivos.

                #### 2. Suite de Índices Espectrales
                1. **NDVI (Índice de Vegetación de Diferencia Normalizada):**
                   $$\\text{NDVI} = \\frac{\\text{B8 (NIR)} - \\text{B4 (Rojo)}}{\\text{B8 (NIR)} + \\text{B4 (Rojo)}}$$
                   Estima la densidad foliar, cobertura vegetal y vigor verde general de la vegetación.
                2. **NDRE (Índice de Borde Rojo de Diferencia Normalizada):**
                   $$\\text{NDRE} = \\frac{\\text{B8 (NIR)} - \\text{B5 (RedEdge 1)}}{\\text{B8 (NIR)} + \\text{B5 (RedEdge 1)}}$$
                   La radiación del Borde Rojo penetra en profundidad dentro de canopeos vegetales densos sin sufrir el efecto de saturación que afecta al NDVI, siendo el mejor estimador de concentración de **clorofila foliar y estado nutricional**.
                3. **NDWI / NDII (Índice de Humedad de Diferencia Normalizada con SWIR):**
                   $$\\text{NDWI} = \\frac{\\text{B8 (NIR)} - \\text{B11 (SWIR 1)}}{\\text{B8 (NIR)} + \\text{B11 (SWIR 1)}}$$
                   Sensible a la absorción de agua en los tejidos vegetales y a la humedad del suelo. Permite discriminar zonas con buena capacidad de retención hídrica de áreas degradadas, arenosas o con limitantes edáficas de profundidad.
                4. **ICA (Índice Compuesto Integrado):**
                   $$\\text{ICA} = 0.40 \\cdot \\text{NDVI} + 0.35 \\cdot \\text{NDRE} + 0.25 \\cdot \\text{NDWI}$$

                #### 3. Algoritmo de Ambientación Multivariado
                En la estrategia **Multivariada Completa**, cada píxel se procesa como un vector tridimensional $\\mathbf{X} = [\\text{NDVI}, \\text{NDRE}, \\text{NDWI}]$. 
                El algoritmo no supervisado **Weka K-Means** en Earth Engine agrupa los píxeles en clusters por similitud espectral. Posteriormente, la plataforma calcula el perfil medio de cada grupo y los reordena de forma ascendente para que el **Ambiente 1 represente el menor nivel de vigor / actividad** y el **Ambiente 3 o 4 el mayor nivel de vigor / actividad**.

                #### 4. Filtrado Espacial Morfológico (Focal Mode)
                La resolución satelital de 10 metros suele capturar ruido local de micro-relieve o artefactos menores a escala de píxel ("efecto sal y pimienta"). 
                El filtro morfológico modal de vecindad de 10 metros **homogeneiza los píxeles aislados reemplazándolos por la clase predominante de su entorno**. 
                Esto permite delimitar **unidades de paisaje continuas, estables y representativas**, facilitando la zonificación de campos, planificación de potreros de pastoreo, manejo de suelos y recorridas de monitoreo territorial.

                #### 5. Visualización Territorial y Navegación a Campo (KML & GeoJSON)
                El archivo `.kml` generado cumple con el estándar abierto OGC KML 2.2 y está optimizado para su uso en diversas plataformas de análisis espacial:
                - **Google Earth Pro:** Permite visualizar el lote en 3D con sus ambientes coloreados según su nivel de vigor / actividad y localizar los puntos de muestreo como chinchetas numeradas.
                - **Sistemas de Información Geográfica (GIS):** Compatible nativamente con QGIS, ArcGIS y gvSIG para análisis espacial avanzado, cruce con cartas de suelos o mapas topográficos.
                - **Dispositivos Móviles y Navegadores GPS:** Los puntos de muestreo pueden cargarse directamente en aplicaciones móviles de campo (Google Maps, OruxMaps, QField, Avenza Maps) o transferirse a receptores GPS de mano (Garmin) para guiar recorridas a campo, calicatas de suelo o muestreos de forraje.

                ---
                #### 🏛️ Créditos Institucionales
                - **Diseño y Metodología:** Lic. Inf. Hernan Elena  
                - **Institución:** Grupo de Recursos Naturales (RRNN) — INTA Estación Experimental Agropecuaria Salta  
                - **Proyecto:** Proyecto PATCHACO  
                - **Tecnologías:** Python, Streamlit, Google Earth Engine API, Folium, GeoPandas, Argenmap (IGN).
                """)

    else:
        st.info("👋 Para comenzar, dibuja un polígono en el mapa inicial o sube un archivo KML/GeoJSON en la barra lateral.")

    # Pie de página institucional
    st.markdown("""
    <div class="footer-credits">
        <b>INTA AMBIENTE</b> • Instituto Nacional de Tecnología Agropecuaria (INTA) • Estación Experimental Agropecuaria Salta<br>
        Diseñado por <b>Lic. Inf. Hernan Elena</b> (Grupo de Recursos Naturales) • Proyecto PATCHACO
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
