"""Recorta Natural Earth 1:50m (dominio público) a la zona del Caribe y lo proyecta a SVG.
Uso: python -I make_map.py <countries50.geojson> <salida.json>"""
import json
import math
import sys

LON0, LON1, LAT0, LAT1 = -90.0, -58.0, 14.0, 34.0
W = 1000.0
KX = math.cos(math.radians(24))
SCALE = W / ((LON1 - LON0) * KX)
H = (LAT1 - LAT0) * SCALE


def xy(lon, lat):
    return round((lon - LON0) * KX * SCALE, 1), round((LAT1 - lat) * SCALE, 1)


def ring_path(ring):
    pts = [xy(lon, lat) for lon, lat in ring]
    out, last = [], None
    for p in pts:  # quita puntos casi repetidos
        if last is None or abs(p[0] - last[0]) + abs(p[1] - last[1]) > 0.8:
            out.append(p)
            last = p
    if len(out) < 3:
        return ""
    return "M" + "L".join(f"{x} {y}" for x, y in out) + "Z"


def in_box(ring):
    return any(LON0 - 2 <= lon <= LON1 + 2 and LAT0 - 2 <= lat <= LAT1 + 2 for lon, lat in ring)


data = json.load(open(sys.argv[1], encoding="utf-8"))
NAMES = {"Cuba": "Cuba", "United States of America": "EE. UU.", "The Bahamas": "Bahamas", "Bahamas": "Bahamas",
         "Mexico": "México", "Haiti": "Haití", "Dominican Rep.": "Rep. Dominicana", "Dominican Republic": "Rep. Dominicana",
         "Jamaica": "Jamaica", "Puerto Rico": "Puerto Rico", "Guatemala": "Guatemala", "Belize": "Belice",
         "Honduras": "Honduras", "Cayman Is.": "Caimán", "Turks and Caicos Is.": "Turcas y Caicos"}
lands = []
for f in data["features"]:
    name = f["properties"].get("NAME") or f["properties"].get("ADMIN")
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    d = "".join(ring_path(poly[0]) for poly in polys if in_box(poly[0]))
    if d:
        lands.append({"name": NAMES.get(name, name), "cuba": name == "Cuba", "d": d})

PLACES = [  # coordenadas aproximadas
    ("habana", "La Habana", 23.13, -82.38, "ciudad"),
    ("mariel", "Mariel", 23.01, -82.75, "puerto"),
    ("san_cristobal", "San Cristóbal (R-12)", 22.71, -83.05, "misiles"),
    ("guanajay", "Guanajay (R-14)", 22.92, -82.69, "misiles"),
    ("sagua", "Sagua la Grande (R-12)", 22.81, -80.08, "misiles"),
    ("remedios", "Remedios (R-14)", 22.49, -79.55, "misiles"),
    ("banes", "Banes: derriban el U-2 de Anderson", 20.96, -75.71, "incidente"),
    ("guantanamo", "Guantánamo (base de EE. UU.)", 19.90, -75.15, "base"),
    ("miami", "Miami", 25.76, -80.19, "ciudad"),
    ("key_west", "Cayo Hueso", 24.56, -81.78, "ciudad"),
]
places = [{"id": i, "label": l, "x": xy(lon, lat)[0], "y": xy(lon, lat)[1], "kind": k} for i, l, lat, lon, k in PLACES]

# Línea de cuarentena: arco a ~500 millas náuticas (≈ 8,3°) del extremo oriental de Cuba (aproximado).
cx, cy = -74.13, 20.25
arc = []
for b in range(-10, 141, 5):  # rumbos de norte a sureste
    r = 500 / 60
    lat = cy + r * math.cos(math.radians(b))
    lon = cx + r * math.sin(math.radians(b)) / math.cos(math.radians(lat))
    arc.append(xy(lon, lat))
quarantine = "M" + "L".join(f"{x} {y}" for x, y in arc)

json.dump({"w": W, "h": round(H, 1), "lands": lands, "places": places, "quarantine": quarantine,
           "source": "Natural Earth 1:50m (dominio público); posiciones y línea de cuarentena aproximadas"},
          open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
print(len(lands), "tierras", round(H), "alto")
