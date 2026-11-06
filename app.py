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
import streamlit.components.v1 as components
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
    page_title="INTA AMBIENTE | PATCHACO",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS_STYLES = """
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
"""

if hasattr(st, "html"):
    st.html(CSS_STYLES)
else:
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. AUTENTICACIÓN GOOGLE EARTH ENGINE (GEE)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Conectando con Google Earth Engine...")
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


def render_folium_map(folium_map, height=500):
    """Renderiza el mapa de Folium de forma nativa como HTML autocontenido (inmune a latencias o caídas de WebSockets)."""
    html_data = folium_map.get_root().render()
    components.html(html_data, height=height)


def clear_map_caches():
    """Limpia los cachés de mapas en session_state para evitar parpadeos y desincronizaciones."""
    st.session_state.pop("cached_prev_html", None)
    st.session_state.pop("cached_prev_aoi", None)
    st.session_state.pop("cached_map_m_html", None)
    st.session_state.pop("cached_map_m_key", None)
    st.session_state.pop("cached_swipe_html", None)
    st.session_state.pop("cached_swipe_key", None)


# ---------------------------------------------------------------------------
# 10. MANUAL DE USUARIO Y METODOLOGÍA CIENTÍFICA
# ---------------------------------------------------------------------------
def render_manual_content():
    """Renderiza la guía paso a paso, marco metodológico y fundamentos científicos."""
    man_tab1, man_tab2, man_tab3 = st.tabs([
        "📋 Guía Paso a Paso de Operación",
        "🔬 Marco Metodológico y Algoritmos",
        "🏛️ Créditos Institucionales"
    ])

    with man_tab1:
        st.markdown(r"""
        ### 📋 Guía Operativa Paso a Paso: ¿Cómo utilizar la plataforma?

        Esta guía describe el flujo de trabajo estándar para ambientar parcelas agrícolas o ganaderas, extraer puntos de muestreo y generar prescripciones operativas:

        ---
        #### 1️⃣ Paso 1: Delimitación del Lote (Área de Interés - AOI)
        - **Opción A (Dibujar en el mapa interactivo):** 
          - Utiliza las herramientas de dibujo ubicadas en la esquina superior izquierda del mapa (icono de **Polígono** o **Rectángulo**).
          - Haz clic en los vértices del potrero y haz doble clic sobre el último punto para cerrar el polígono.
          - *Geoposicionamiento Rápido:* En la barra lateral dispones del módulo desplegable **📍 Geoposicionar / Ir a Coordenadas**. Puedes ingresar coordenadas en formato **Decimal** (ej: `-24.9015, -65.4820`) o **Sexagesimal DMS** (ej: `24° 54' 05" S, 65° 28' 55" O`) y presionar el botón correspondiente para centrar el mapa automáticamente sobre tu campo antes de trazarlo.
        - **Opción B (Subir archivo vectorial existente):**
          - Selecciona la opción **📁 Subir KML / GeoJSON** en la barra lateral y arrastra tu archivo `.kml`, `.kmz`, `.geojson` o `.json`. La plataforma extraerá la geometría y centrará la vista de inmediato.
        - **Límite Operativo de Seguridad (10.000 ha):**
          - La plataforma admite parcelas de hasta **10.000 hectáreas** (100 km²) por análisis. Si se supera este valor, el sistema advertirá al usuario para que acote la geometría o la divida en potreros menores, asegurando respuestas ágiles y evitando saturar cuotas en Google Earth Engine.

        ---
        #### 2️⃣ Paso 2: Filtro Temporal Sentinel-2 y Nubes
        - **Rango de Fechas (Inicio y Fin):**
          - Por defecto se sugiere un intervalo de 90 días hacia atrás desde la fecha actual.
          - Se recomienda seleccionar una ventana temporal de **60 a 90 días** coincidente con el período crítico del cultivo (ej: floración / llenado de granos en cultivos de grano) o la época de máxima actividad forrajera en pasturas y pastizales naturales.
        - **Cobertura Máxima de Nubes (%):**
          - Permite descartar escenas con nubosidad general elevada (por defecto 15%).
          - Además, la plataforma aplica la máscara píxel a píxel **Scene Classification Layer (SCL)** provista por la ESA a 10-20 m, descartando sombras, cirros y nubes residuales de cada imagen.
        - **Composición Multitemporal por Mediana:**
          - Todos los píxeles despejados dentro de la ventana de fechas se integran mediante la **mediana estadística**, eliminando cualquier distorsión por destellos o anomalías atmosféricas puntuales.

        ---
        #### 3️⃣ Paso 3: Estrategia Agronómica y Criterio de Segmentación
        - **Criterio de Segmentación:**
          - **🌟 Multivariada Completa (NDVI + NDRE + NDWI):** Integra en simultáneo biomasa foliar, clorofila y contenido hídrico/humedad de suelo en un espacio tridimensional mediado por K-Means. Es la opción recomendada para ambientaciones de alta precisión.
          - **🌿 Monovariables (NDVI, NDRE, NDWI o ICA):** Para análisis orientados a una variable específica.
        - **Algoritmo para opciones Monovariables:**
          - *📊 Cuantiles Agronómicos (Percentiles):* Divide el lote en áreas de **superficie equivalente** (terciles al 33% o cuartiles al 25%).
          - *🤖 Weka K-Means (Clustering GEE):* Agrupa por **afinidad radiométrica natural** respetando la extensión real de los quiebres o limitantes del lote.
        - **Cantidad de Zonas de Manejo (3 o 4):**
          - *3 Zonas:* Ambiente 1 (Baja), Ambiente 2 (Media), Ambiente 3 (Alta).
          - *4 Zonas:* Ambiente 1 (Baja), Ambiente 2 (Media), Ambiente 3 (Media-Alta), Ambiente 4 (Alta).
        - **Aplicar Suavizado Espacial (Focal Mode 10m):**
          - Recomendado mantener activado. Reemplaza píxeles aislados por la moda predominante de su entorno, transformando mapas pixelados en **paños continuos y limpios**, listos para maquinaria agrícola y dosificación variable de insumos.

        ---
        #### 4️⃣ Paso 4: Puntos de Muestreo a Campo (GPS)
        - **Selección de Puntos por Ambiente:**
          - Define con el selector cuántos sitios de control deseas relevar dentro de cada zona (de 1 a 15, con tope dinámico adaptado al tamaño del potrero).
          - El sistema distribuye los puntos en el **interior de los polígonos representativos** maximizando la distancia euclidiana entre ellos para evitar aglomeraciones.

        ---
        #### 5️⃣ Paso 5: Procesamiento y Exploración Visual
        - Presiona **🚀 Iniciar Procesamiento de Ambientación** (o desde la barra lateral).
        - **Pestaña 🗺️ Visualizador Espacial:**
          - *Modo Multicapa Estándar:* Enciende o apaga capas temáticas (Google Satélite, Argenmap IGN, Ambientes, Puntos de Muestreo, NDVI, NDRE, NDWI, RGB y CIR) desde el selector de capas superior derecho.
          - *Modo Cortina Deslizante (SWIPE):* Compara interactivamente dos capas deslizando la barra vertical divisoria sobre el mapa (ej: Satélite de fondo vs Ambientación resultante).
        - **Pestaña 📈 Curva Fenológica Temporal:**
          - Analiza la serie de tiempo multitemporal del NDVI para cada uno de los ambientes clasificados, evaluando la estacionalidad y estabilidad productiva interanual o del ciclo del cultivo.
        - **Pestaña 📍 Puntos de Muestreo (GPS):**
          - Tabla detallada con cada punto identificado por ambiente (`A1_P1`, `A2_P1`, etc.), superficie del paño y coordenadas en formato **Decimal** y **Sexagesimal (DMS)**.
        - **Pestaña 📊 Diagnóstico y Estadísticas:**
          - Desglose de superficie en hectáreas y porcentaje ocupado por cada ambiente, junto a sus valores promedios espectrales.

        ---
        #### 6️⃣ Paso 6: Exportación de Archivos para Campo y SIG
        - **💾 Descargar KML OGC:** Archivo estructurado con polígonos coloreados translúcidos y Waypoints con fichas descriptivas completas, listo para abrir en **Google Earth Pro**, cargar en navegadores **GPS Garmin** o visualizar en aplicaciones móviles de campo (**OruxMaps**, **QField**, **Avenza Maps**, **Google Maps**).
        - **💾 Descargar GeoJSON:** Capa vectorial con atributos alfanuméricos estandarizados para análisis en **QGIS**, **ArcGIS** o scripts de geoprocesamiento.
        - **💾 Descargar CSV de Puntos GPS:** Planilla tabular para importar a Excel, diseñar planillas de calicatas o guiar muestreadores de suelo y biomasa.
        """)

    with man_tab2:
        st.markdown(r"""
        ### 🔬 Marco Metodológico y Fundamentos Teóricos

        #### 1. Fuentes Satelitales y Calibración Radiométrica
        **INTA AMBIENTE** procesa la constelación **Sentinel-2A y Sentinel-2B** de la Agencia Espacial Europea (ESA), a través de la colección **`COPERNICUS/S2_SR_HARMONIZED`** en Google Earth Engine:
        - **Resolución Espacial:** 10 metros nativos en el visible e Infrarrojo Cercano (NIR); 20 metros en Red Edge e Infrarrojo de Onda Corta (SWIR), remuestreados internamente a 10 metros para máxima compatibilidad geométrica.
        - **Nivel de Procesamiento:** Nivel 2A (Reflectancia de Superficie en el fondo de la atmósfera - BOA), corregido de aerosoles atmosféricos.
        - **Máscara de Nubes Píxel a Píxel (SCL):** Descarta automáticamente sombras de nubes (clase 3), nubes de media y alta probabilidad (clases 8 y 9), cirros finos (clase 10) y nieve (clase 11).
        - **Reducción Temporal por Mediana:** Evalúa el vector temporal de cada píxel despejado y toma el valor mediano, garantizando total inmunidad frente a destellos de suelo, sombras temporales o nubosidades dispersas.

        ---
        #### 2. Suite de Índices Espectrales Agronómicos
        1. **NDVI (Índice de Vegetación de Diferencia Normalizada):**
           $$\text{NDVI} = \frac{\text{B8 (NIR)} - \text{B4 (Rojo)}}{\text{B8 (NIR)} + \text{B4 (Rojo)}}$$
           Estima el vigor verde, cobertura vegetal y densidad foliar activa.
        2. **NDRE (Índice de Borde Rojo de Diferencia Normalizada):**
           $$\text{NDRE} = \frac{\text{B8 (NIR)} - \text{B5 (RedEdge 1)}}{\text{B8 (NIR)} + \text{B5 (RedEdge 1)}}$$
           La radiación del Borde Rojo penetra en profundidad dentro de canopeos vegetales densos sin saturarse como el NDVI, siendo el mejor estimador de concentración de **clorofila foliar y nutrición nitrogenada**.
        3. **NDWI (Índice de Agua de Diferencia Normalizada con SWIR):**
           $$\text{NDWI} = \frac{\text{B8 (NIR)} - \text{B11 (SWIR 1)}}{\text{B8 (NIR)} + \text{B11 (SWIR 1)}}$$
           Sensible al contenido de agua en tejidos vegetales y a la humedad del suelo. Permite discriminar sectores con buena retención hídrica de zonas con limitantes edáficas de profundidad, tosca o arenas.
        4. **ICA (Índice Compuesto Integrado 40/35/25):**
           $$\text{ICA} = 0.40 \cdot \text{NDVI} + 0.35 \cdot \text{NDRE} + 0.25 \cdot \text{NDWI}$$
           Sintetiza en un único valor ponderado la cobertura foliar, la actividad fotosintética y el estado hídrico.

        ---
        #### 3. Algoritmo de Ambientación Multivariado
        En la estrategia **Multivariada Completa**, cada píxel se procesa como un vector tridimensional:
        $$\mathbf{X} = [\text{NDVI}, \text{NDRE}, \text{NDWI}]$$
        El algoritmo **Weka K-Means** en Earth Engine agrupa los píxeles en clusters minimizando la varianza cuadrática intra-cluster en el espacio tridimensional. Posteriormente, la plataforma calcula el perfil medio de cada grupo y los reordena de forma ascendente para que el **Ambiente 1 represente el menor nivel de vigor / actividad** y el **Ambiente 3 o 4 el mayor nivel de vigor / actividad**.

        ---
        #### 4. 🔀 Clasificación Monovariable: ¿Cuantiles Agronómicos o Weka K-Means?
        Cuando se selecciona una estrategia **Monovariable** (basada en un único índice como NDVI, NDRE, NDWI o ICA), la plataforma permite elegir entre dos metodologías matemáticas de segmentación:

        ##### A) 📊 Cuantiles Agronómicos (Percentiles)
        - **Principio Matemático:** Segmenta el histograma de frecuencias de los valores del índice en intervalos que abarcan **exactamente la misma proporción de superficie o píxeles**:
          - Para **3 Zonas:** Percentiles 33% y 66% (Terciles). Cada ambiente agrupará aproximadamente un tercio (33,3%) de la superficie del lote.
          - Para **4 Zonas:** Percentiles 25%, 50% y 75% (Cuartiles). Cada ambiente agrupará aproximadamente un cuarto (25%) de la superficie del lote.
        - **Significado Agronómico:**
          - Realiza una clasificación **estrictamente relativa** dentro de la parcela.
          - El Ambiente 1 siempre corresponderá al 25% o 33% inferior de ese lote específico, sin importar si el lote en general tiene un nivel de vigor alto o bajo.
        - **¿Cuándo utilizarlo?**
          - Cuando se busca **equilibrio de superficie** para la gestión logística (ej: dividir un potrero en partes iguales para rotaciones de pastoreo o compras homogéneas de semillas y fertilizantes).
          - Para muestreos a campo con igual esfuerzo de muestreo por estrato de superficie.

        ##### B) 🤖 Weka K-Means (Clustering GEE)
        - **Principio Matemático:** Algoritmo no supervisado de *Machine Learning* que localiza centroides en el espacio radiométrico unidimensional, agrupando los píxeles según su cercanía matemática natural, **sin imponer ningún porcentaje de superficie obligatorio**.
        - **Significado Agronómico:**
          - Identifica **quiebres y discontinuidades espaciales reales** en el terreno.
          - Si en el potrero existe una limitante severa (ej: un bajo salino, un sector anegable, un manto rocoso o un manchón de erosión) que abarca únicamente el **8% o el 14% del área**, K-Means creará un ambiente propio para ese sector con su dimensión real exacta. Por el contrario, los cuantiles forzarían a ese sector a mezclarse con sectores mejores para completar obligatoriamente el 25% o 33%.
        - **¿Cuándo utilizarlo?**
          - Cuando el objetivo es la **detección precisa de limitantes edáficas, fallas de implantación o manchones asimétricos**.
          - Para prescripciones variables donde los ambientes tienen dimensiones naturalmente desiguales.

        ---
        #### 5. Filtrado Espacial Morfológico (Focal Mode 10m)
        La resolución nativa de 10 m suele registrar píxeles individuales anómalos por micro-relieve o artefactos menores ("efecto sal y pimienta"). 
        El filtro morfológico modal reemplaza los píxeles aislados por la clase predominante de su entorno inmediato. Esto genera **unidades de paisaje continuas, estables y operativas**, facilitando el tránsito de maquinaria y la demarcación de alambrados o lotes de manejo.

        ---
        #### 6. Algoritmo de Muestreo Interior Voraz (Greedy Distance)
        Para cada polígono de ambiente delimitado:
        1. Se genera una grilla interna de puntos candidatos filtrada estrictamente por contención geométrica interior (`target.contains(Point)`).
        2. Se aplica una selección voraz (*greedy selection*) que maximiza la distancia euclidiana mínima respecto a los puntos ya seleccionados:
           $$P_{i} = \arg\max_{c \in \text{Candidatos}} \left( \min_{s \in \text{Seleccionados}} d(c, s) \right)$$
        3. Esto asegura que los puntos de control estén representativamente esparcidos por el potrero y nunca caigan en bordes, callejones o zonas de transición.
        """)

    with man_tab3:
        st.markdown(r"""
        ### 🏛️ Créditos Institucionales y Equipo de Desarrollo

        - **Diseño Metodológico y Desarrollo de Software:**  
          👨‍💻 **Lic. Inf. Hernan Elena**  
          *Grupo de Recursos Naturales (RRNN)*  
          **Instituto Nacional de Tecnología Agropecuaria (INTA)**  
          *Estación Experimental Agropecuaria Salta (EEA Salta)*  

        - **Programa Institucional:**  
          🌿 **Proyecto PATCHACO** (Planificación Agroecológica Territorial del Chaco y Áreas de Influencia)

        - **Pila Tecnológica:**  
          Python 3, Streamlit, Google Earth Engine API (`earthengine-api`), Folium, GeoPandas, Shapely, Plotly, Argenmap (Instituto Geográfico Nacional - IGN Argentina).

        - **Uso y Citación:**  
          Plataforma abierta y pública orientada a productores, asesores técnicos, investigadores y extensionistas para el manejo por ambientes y la intensificación ecológica de los recursos naturales.
        """)


@st.dialog("📖 Manual de Usuario y Metodología Científica", width="large")
def show_manual_dialog():
    """Diálogo emergente (popup modal) con el manual y la metodología."""
    render_manual_content()


# ---------------------------------------------------------------------------
# 11. APLICACIÓN PRINCIPAL (STREAMLIT UI)
# ---------------------------------------------------------------------------
def main():
    # Encabezado institucional limpio (el logo permanece en la barra lateral)
    st.title("🌾 INTA AMBIENTE")
    st.markdown("##### Plataforma de Ambientación Multivariada, Fenología Satelital y Muestreo a Campo")
    st.caption("🏛️ **Proyecto PATCHACO** • Grupo de Recursos Naturales (RRNN) • **INTA EEA Salta**")

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
        clear_map_caches()

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
                if st.session_state.get("current_aoi") != geom:
                    st.session_state["current_aoi"] = geom
                    clear_map_caches()
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
            clear_map_caches()
            st.rerun()

    # Separador y botón de acceso al Manual de Usuario y Metodología en Diálogo Emergente (Popup)
    st.sidebar.markdown("---")
    if st.sidebar.button("📖 Manual y Metodología", use_container_width=True, help="Abre la guía paso a paso y los fundamentos científicos en un diálogo emergente"):
        show_manual_dialog()

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

        try:
            map_output = st_folium(
                m_draw,
                use_container_width=True,
                height=530,
                key="draw_map",
                returned_objects=["last_active_drawing"]
            )
        except Exception:
            render_folium_map(m_draw, height=530)
            map_output = None
        
        if map_output and map_output.get("last_active_drawing"):
            geom_dict = map_output["last_active_drawing"].get("geometry")
            if geom_dict:
                geom, err = parse_geojson_to_shapely(geom_dict)
                if not err:
                    st.session_state["current_aoi"] = geom
                    st.session_state["aoi_name"] = "Lote_Dibujado"
                    clear_map_caches()
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

        # 1. EJECUCIÓN DEL MOTOR DE PROCESAMIENTO (SI FUE DISPARADO)
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
                    clear_map_caches()
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error durante el procesamiento: {str(ex)}")
                    import traceback
                    st.code(traceback.format_exc())


        # 2. ESTADO A: AÚN NO SE PROCESÓ -> VISTA PREVIA Y CONFIRMACIÓN DE LOTE
        if st.session_state["results"] is None:
            st.markdown("### 📍 Vista Previa del Lote Delimitado")
            st.info(f"Parcela: **{st.session_state['aoi_name']}** — Superficie: **{lote_ha} hectáreas**. Revisa los límites y presiona el botón para procesar la ambientación.")

            # Caché del HTML del mapa de vista previa para evitar parpadeos y recargas
            # al interactuar con los controles de la barra lateral (puntos, suavizado, etc.)
            aoi_signature = f"{shapely_aoi.wkt}_{st.session_state.get('aoi_name', '')}"
            if st.session_state.get("cached_prev_aoi") != aoi_signature or "cached_prev_html" not in st.session_state:
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
                st.session_state["cached_prev_html"] = m_prev.get_root().render()
                st.session_state["cached_prev_aoi"] = aoi_signature

            components.html(st.session_state["cached_prev_html"], height=480)

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
                clear_map_caches()
                st.rerun()


        # 3. ESTADO B: YA SE PROCESÓ -> VISUALIZADOR ESPACIAL Y REPORTES
        else:
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

            # Barra superior de navegación y acciones de resultados
            col_res_header, col_res_actions = st.columns([2, 1])
            with col_res_header:
                st.markdown(f"### 🌾 Resultados de Ambientación: **{st.session_state['aoi_name']}**")
                st.caption(f"Superficie: **{lote_ha} ha** • Estrategia: **{res_mode}** • **{res_num_zones} Zonas de Manejo**")
            with col_res_actions:
                act_c1, act_c2 = st.columns(2)
                if act_c1.button("🔙 Ajustar Lote", use_container_width=True, help="Vuelve a la vista previa del contorno para cambiar parámetros o fechas"):
                    st.session_state["results"] = None
                    st.rerun()
                if act_c2.button("🔄 Re-procesar", type="primary", use_container_width=True, help="Recalcula la ambientación con los parámetros actuales de la barra lateral"):
                    st.session_state["trigger_process"] = True
                    st.rerun()

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

                    swipe_key = f"{id(res)}_{swipe_left_name}_{swipe_right_name}"
                    if st.session_state.get("cached_swipe_key") != swipe_key or "cached_swipe_html" not in st.session_state:
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
                        st.session_state["cached_swipe_html"] = m_swipe.get_root().render()
                        st.session_state["cached_swipe_key"] = swipe_key

                    components.html(st.session_state["cached_swipe_html"], height=620)

                # --- MODO 2: MULTICAPA ESTÁNDAR ---
                else:
                    st.caption("Usa el menú de capas (arriba a la derecha) para alternar entre Argenmap, Google Satélite, Ambientación, Puntos de Muestreo e Índices.")

                    m_key = f"{id(res)}_{res_num_zones}_{len(gdf_points)}"
                    if st.session_state.get("cached_map_m_key") != m_key or "cached_map_m_html" not in st.session_state:
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
                        st.session_state["cached_map_m_html"] = m.get_root().render()
                        st.session_state["cached_map_m_key"] = m_key

                    components.html(st.session_state["cached_map_m_html"], height=620)

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
                render_manual_content()

    else:
        st.info("👋 Para comenzar, dibuja un polígono en el mapa inicial o sube un archivo KML/GeoJSON en la barra lateral.")

    # Pie de página institucional
    st.markdown("---")
    st.caption("🏛️ **INTA AMBIENTE** • Instituto Nacional de Tecnología Agropecuaria (INTA) • Estación Experimental Agropecuaria Salta  \n👨‍💻 Diseñado por **Lic. Inf. Hernan Elena** (Grupo de Recursos Naturales) • Proyecto PATCHACO")


if __name__ == "__main__":
    main()
