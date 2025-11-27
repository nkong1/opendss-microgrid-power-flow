import re
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString


# --- Parse DSS elements (Lines, Loads, Transformers) ---
def parse_dss_file(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('!')]
    pattern = re.compile(r'new\s+(\w+)\.(\S+)\s+(.*)', re.IGNORECASE)
    records = []
    for l in lines:
        match = pattern.match(l)
        if not match:
            continue
        dtype, name, rest = match.groups()
        attrs = {'type': dtype.lower(), 'name': name}
        for kv in re.findall(r'(\w+)=([^\s]+)', rest):
            k, v = kv
            attrs[k.lower()] = v
        records.append(attrs)
    return pd.DataFrame(records)


# --- Parse BusCoords + voltages ---
def parse_buscoords(path):
    coords = {}
    voltages = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            # Parse SetBusXY: SetBusXY [busname] [lat] [lon]
            if line.lower().startswith('setbusxy'):
                parts = re.split(r'[,\s]+', line)
                if len(parts) >= 4:
                    bus = parts[1]
                    try:
                        lat, lon = float(parts[2]), float(parts[3])  # Correct order for QGIS
                        coords[bus] = (lat, lon)
                    except ValueError:
                        continue

            # Parse SetkVBase: SetkVBase [busname] kVLL=xx
            elif line.lower().startswith('setkvbase'):
                match = re.search(r'setkvbase\s+([\w\-]+).*?kvll=([\d\.]+)', line, re.IGNORECASE)
                if match:
                    bus_id, kvll = match.groups()
                    voltages[bus_id] = float(kvll)

    # Merge both sets
    buses = {
        b: {'lat': coords[b][0], 'lon': coords[b][1], 'kvll': voltages.get(b, None)}
        for b in coords
    }
    return buses


# --- Helper for safely creating geometry Points ---
def safe_point(bus_id, bus_coords):
    if bus_id in bus_coords:
        data = bus_coords[bus_id]
        if 'lat' in data and 'lon' in data:
            return Point(data['lon'], data['lat'])  # ensure lon, lat order
    return None


# --- Main GeoPackage Builder ---
def opendss_to_gpkg(loads_path, lines_path, transformers_path, buscoords_path, output_gpkg='OpenDSS_Export.gpkg'):
    bus_coords = parse_buscoords(buscoords_path)

    # --- LOADS ---
    df_loads = parse_dss_file(loads_path)
    if 'bus1' not in df_loads.columns:
        df_loads['bus1'] = None
    df_loads['geometry'] = df_loads['bus1'].map(
        lambda b: safe_point(b.split('.')[0], bus_coords) if isinstance(b, str) else None
    )
    gdf_loads = gpd.GeoDataFrame(df_loads, geometry='geometry', crs='EPSG:4326')

    # --- LINES ---
    df_lines = parse_dss_file(lines_path)
    for col in ['bus1', 'bus2']:
        if col not in df_lines.columns:
            df_lines[col] = None

    # Add voltage attributes for each line based on its connected buses
    def get_line_voltage(row):
        if isinstance(row['bus1'], str):
            bus1 = row['bus1'].split('.')[0]
            if bus1 in bus_coords and bus_coords[bus1].get('kvll'):
                return bus_coords[bus1]['kvll']
        return None

    df_lines['kvll'] = df_lines.apply(get_line_voltage, axis=1)  # derive from bus1

    # Create geometry safely using bus coordinates
    df_lines['geometry'] = df_lines.apply(
        lambda r: LineString([
            (bus_coords[r['bus1'].split('.')[0]]['lon'], bus_coords[r['bus1'].split('.')[0]]['lat']),
            (bus_coords[r['bus2'].split('.')[0]]['lon'], bus_coords[r['bus2'].split('.')[0]]['lat'])
        ]) if isinstance(r['bus1'], str) and isinstance(r['bus2'], str)
        and r['bus1'].split('.')[0] in bus_coords
        and r['bus2'].split('.')[0] in bus_coords else None,
        axis=1
    )

    gdf_lines = gpd.GeoDataFrame(df_lines, geometry='geometry', crs='EPSG:4326')


    # --- TRANSFORMERS ---
    df_tx = parse_dss_file(transformers_path)
    df_tx = df_tx[df_tx['type'] == 'transformer']
    if 'buses' not in df_tx.columns:
        df_tx['buses'] = None

    def transform_point(row):
        b = row['buses']
        if isinstance(b, str):
            buses = re.findall(r'[\w\-]+', b)
            if buses:
                bus_name = buses[0].split('.')[0]
                return safe_point(bus_name, bus_coords)
        return None

    df_tx['geometry'] = df_tx.apply(transform_point, axis=1)
    gdf_tx = gpd.GeoDataFrame(df_tx, geometry='geometry', crs='EPSG:4326')

    # --- BUSES (with voltages) ---
    df_buses = pd.DataFrame([
        (b, d['lat'], d['lon'], d['kvll']) for b, d in bus_coords.items()
    ], columns=['bus', 'lat', 'lon', 'kvll'])
    df_buses['geometry'] = df_buses.apply(lambda r: Point(r['lon'], r['lat']), axis=1)
    gdf_buses = gpd.GeoDataFrame(df_buses, geometry='geometry', crs='EPSG:4326')

    # --- EXPORT GeoPackage ---
    gdf_buses.to_file(output_gpkg, layer='buses', driver='GPKG')
    gdf_lines.to_file(output_gpkg, layer='lines', driver='GPKG')
    gdf_tx.to_file(output_gpkg, layer='transformers', driver='GPKG')
    gdf_loads.to_file(output_gpkg, layer='loads', driver='GPKG')

    print(f"✅ Export complete — all geometry and voltage attributes written to: {output_gpkg}")
    return output_gpkg


# --- Run ---
if __name__ == '__main__':
    opendss_to_gpkg('Loads.dss', 'Lines.dss', 'Transformers.dss', 'BusCoords.dss')
