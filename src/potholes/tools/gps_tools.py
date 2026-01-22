import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pyproj import Transformer


def filter_by_distance(df, min_meters=2):
    kept = [df.iloc[0]]
    last = (df.iloc[0]["latitude"], df.iloc[0]["longitude"])

    for _, r in df.iloc[1:].iterrows():
        current = (r["latitude"], r["longitude"])
        if geodesic(last, current).meters >= min_meters:
            kept.append(r)
            last = current

    return pd.DataFrame(kept)


def get_run_data(df: pd.DataFrame):
    gps = df.dropna(subset=["latitude", "longitude"]).copy()
    gps = gps.sort_values("timestamp")

    is_new_fix = (gps["latitude"].ne(gps["latitude"].shift()) |
                  gps["longitude"].ne(gps["longitude"].shift()))
    gps = gps[is_new_fix]

    transformer = Transformer.from_crs(
        "EPSG:4326",  # lat/lon
        "EPSG:32630",  # UTM 30N
        always_xy=True
    )

    gps["x_m"], gps["y_m"] = transformer.transform(
        gps["longitude"].values,
        gps["latitude"].values
    )

    t = gps["timestamp"].values.astype("datetime64[ns]")
    dt = np.diff(t).astype("timedelta64[ns]").astype(float) * 1e-9

    dx = np.diff(gps["x_m"].values)
    dy = np.diff(gps["y_m"].values)

    vx = dx / dt
    vy = dy / dt

    gps = gps.iloc[1:].copy()
    gps["vx"] = vx
    gps["vy"] = vy
    gps["speed_mps"] = np.sqrt(vx ** 2 + vy ** 2)
    gps["speed_kmh"] = gps["speed_mps"] * 3.6

    dvx = np.diff(vx)
    dvy = np.diff(vy)
    dt2 = dt[1:]

    ax = dvx / dt2
    ay = dvy / dt2

    gps = gps.iloc[1:].copy()
    gps["ax"] = ax
    gps["ay"] = ay
    gps["acc_mps2"] = np.sqrt(ax ** 2 + ay ** 2)

    return gps

def plot_route(df: pd.DataFrame):
    LABEL_COLORS = {
        "pothole": "red",
        "other": "blue",
        "manhole": "purple",
        "speed_bump": "green",
    }

    gps = (
        df.dropna(subset=["latitude", "longitude"])
          .sort_values("timestamp")
    )

    gps_unique = gps.loc[
        (gps["latitude"].diff().ne(0)) |
        (gps["longitude"].diff().ne(0))
    ]

    gps_unique = filter_by_distance(gps_unique, min_meters=3)

    center = [gps_unique["latitude"].mean(), gps_unique["longitude"].mean()]

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    folium.PolyLine(
        gps_unique[["latitude", "longitude"]].values.tolist(),
        weight=3
    ).add_to(m)

    labeled = gps[gps["label"].notna()]
    for _, r in labeled.iterrows():
        label = str(r["label"])
        color = LABEL_COLORS.get(label, "gray")  # fallback

        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=label
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index:9999;
        background-color:white;
        padding:10px;   
        border:2px solid grey;
        font-size:14px;
    ">
    <b>Labels</b><br>
    <span style="color:red;">●</span> Pothole<br>
    <span style="color:green;">●</span> Speed bump<br>
    <span style="color:purple;">●</span> Manhole<br>
    <span style="color:blue;">●</span> Other<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


if __name__ == "__main__":
    # session_file = R"data/Data_20260108/session_STBPRO3@71E957_20260108_161834.csv"
    session_file = R"data/Data_20260108/session_STBPRO3@71E957_20260108_164556.csv"
    data = pd.read_csv(session_file, parse_dates=["timestamp"])
    map = plot_route(data)
    map.save("output/route_test_2.html")