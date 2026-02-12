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
    LABEL_DISPLAY = {
        "pothole": "Pothole",
        "speed_bump": "Speed bump",
        "manhole": "Manhole",
        "other": "Other",
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

    labeled = gps[gps["label"].notna()].copy()

    label_counts = labeled["label"].astype(str).value_counts().to_dict()

    legend_order = ["pothole", "speed_bump", "manhole", "other"]
    for label in label_counts:
        if label not in LABEL_COLORS:
            LABEL_COLORS[label] = "gray"
        if label not in LABEL_DISPLAY:
            LABEL_DISPLAY[label] = str(label).replace("_", " ").title()
        if label not in legend_order:
            legend_order.append(label)

    # Group markers by label so each group can be toggled from the custom legend.
    label_groups = {}
    for label in legend_order:
        group = folium.FeatureGroup(name=label, show=True)
        group.add_to(m)
        label_groups[label] = group

    for _, r in labeled.iterrows():
        label = str(r["label"])
        color = LABEL_COLORS.get(label, "gray")  # fallback
        elapsed_value = pd.to_numeric(r.get("elapsed"), errors="coerce") / 1e9
        elapsed_text = "N/A" if pd.isna(elapsed_value) else f"{elapsed_value:.3f} s"
        popup_html = (
            f"<b>Label:</b> {label}<br>"
            f"<b>Elapsed:</b> {elapsed_text}<br>"
            f"<b>Latitude:</b> {r['latitude']:.7f}<br>"
            f"<b>Longitude:</b> {r['longitude']:.7f}"
        )

        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(label_groups[label])

    legend_rows = "".join(
        (
            f'<div class="legend-item" data-layer="{label_groups[label].get_name()}">'
            f'<span style="color:{LABEL_COLORS[label]};">●</span> '
            f'{LABEL_DISPLAY[label]} ({label_counts.get(label, 0)})'
            f"</div>"
        )
        for label in legend_order
    )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index:9999;
        background-color:white;
        padding:10px;
        border:2px solid grey;
        font-size:14px;
        min-width: 170px;
    ">
    <b>Labels</b><br>
    {legend_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    legend_style = """
    <style>
      .legend-item {
        cursor: pointer;
        user-select: none;
        margin-top: 4px;
      }
      .legend-item.inactive {
        opacity: 0.35;
      }
    </style>
    """
    m.get_root().html.add_child(folium.Element(legend_style))

    toggle_script = f"""
    (function() {{
      function setupLegendToggles(attempt) {{
        var mapRef = window["{m.get_name()}"];
        if (!mapRef) {{
          if (attempt < 40) {{
            setTimeout(function() {{ setupLegendToggles(attempt + 1); }}, 100);
          }}
          return;
        }}

        var legendItems = document.querySelectorAll(".legend-item");
        legendItems.forEach(function(item) {{
          var layerName = item.getAttribute("data-layer");
          var layerRef = window[layerName];
          if (!layerRef) {{
            return;
          }}

          item.classList.toggle("inactive", !mapRef.hasLayer(layerRef));
          item.addEventListener("click", function() {{
            if (mapRef.hasLayer(layerRef)) {{
              mapRef.removeLayer(layerRef);
              item.classList.add("inactive");
            }} else {{
              mapRef.addLayer(layerRef);
              item.classList.remove("inactive");
            }}
          }});
        }});
      }}

      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", function() {{
          setupLegendToggles(0);
        }});
      }} else {{
        setupLegendToggles(0);
      }}
    }})();
    """
    m.get_root().script.add_child(folium.Element(toggle_script))

    return m


if __name__ == "__main__":
    # session_file = R"data/Data_20260108/session_STBPRO3@71E957_20260108_161834.csv"
    session_file = R"data/Data_20260108/session_STBPRO3@71E957_20260108_164556.csv"
    data = pd.read_csv(session_file, parse_dates=["timestamp"])
    map = plot_route(data)
    map.save("output/route_test_2.html")
