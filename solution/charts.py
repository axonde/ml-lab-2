import numpy as np
import pandas as pd
import plotly.express as px

from research import AREA_ORDER, LIGHT_ORDER, TOP_CATEGORIES, UNLIT, fmt_int, share_table

BLUE = "#2a78d6"
ACCENT = "#B33440"
GRAY = "#b4b2a9"
AREA_COLORS = {"Благовещенск": "#2a78d6", "Другие города": "#1baf7a", "Районы": "#eb6834"}
SEVERITY_ORDER = ["Легкий", "Тяжёлый", "С погибшими"]
SEVERITY_COLORS = {"Легкий": "#3579B8", "Тяжёлый": "#C18A16", "С погибшими": "#B33440"}
KIND_COLORS = {
    "Столкновение": "#2a78d6", "Наезд на пешехода": "#eb6834",
    "Съезд с дороги или опрокидывание": "#1baf7a", "Другое": GRAY,
}
COUNT_COLORS = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
SHARE_COLORS = ["#fcbba1", "#ef3b2c", "#67000d"]
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
SHARE = "ДТП с погибшими, %"
MAP_NOTE = "Подложка: © OpenStreetMap contributors, https://www.openstreetmap.org/copyright."


def with_errors(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(plus=table["high"] - table["share"], minus=table["share"] - table["low"])


def monthly_chart(current: pd.DataFrame, start, end):
    counts = current.set_index("datetime").resample("MS").size()
    months = pd.date_range(pd.Timestamp(start).to_period("M").start_time,
                           pd.Timestamp(end).to_period("M").start_time, freq="MS")
    counts = counts.reindex(months, fill_value=0)
    table = pd.DataFrame({
        "month": months,
        "ДТП за месяц": counts.to_numpy(),
        "Среднее за 12 месяцев": counts.rolling(12).mean().to_numpy(),
    })
    fig = px.line(table, x="month", y=["ДТП за месяц", "Среднее за 12 месяцев"],
                  color_discrete_sequence=["#86b6ef", "#104281"],
                  labels={"month": "Месяц", "value": "Число ДТП", "variable": ""})
    fig.update_layout(hovermode="x unified", legend={"orientation": "h", "y": 1.12})
    return fig


def category_chart(current: pd.DataFrame):
    kind = current["category"].where(current["category"].isin(TOP_CATEGORIES), "Прочие виды")
    table = current.assign(kind=kind).groupby("kind").agg(n=("id", "size"), fatal=("fatal", "mean"))
    table = table.reset_index().sort_values("n")
    table["percent"] = table["n"] / table["n"].sum() * 100
    table["fatal"] = table["fatal"] * 100
    table["label"] = [f"{fmt_int(n)} ({p:.0f}%)" for n, p in zip(table["n"], table["percent"])]
    fig = px.bar(table, x="n", y="kind", orientation="h", text="label",
                 color_discrete_sequence=[BLUE], hover_data={"fatal": ":.1f", "label": False},
                 labels={"n": "Число ДТП", "kind": "", "fatal": SHARE})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, table["n"].max() * 1.25])
    return fig


def victims_chart(current: pd.DataFrame):
    victims = current["victims"].clip(upper=6).astype(int)
    table = victims.value_counts().reindex(range(1, 7), fill_value=0).rename("n").reset_index()
    table["label"] = table["victims"].astype(str).replace({"6": "6 и больше"})
    table["percent"] = table["n"] / table["n"].sum() * 100
    fig = px.bar(table, x="label", y="percent", text=table["percent"].map("{:.1f}%".format),
                 color_discrete_sequence=[BLUE], hover_data={"n": True, "label": False},
                 labels={"label": "Пострадавших в одном ДТП (раненые + погибшие), чел.",
                         "percent": "% ДТП в срезе", "n": "Число ДТП"})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(type="category")
    return fig


def heatmap_chart(current: pd.DataFrame):
    table = pd.crosstab(current["weekday"], current["hour"])
    table = table.reindex(index=range(7), columns=range(24), fill_value=0)
    fig = px.imshow(table.to_numpy(), x=list(range(24)), y=WEEKDAYS, aspect="auto",
                    color_continuous_scale="Blues",
                    labels={"x": "Час начала ДТП", "y": "День недели", "color": "Число ДТП"})
    fig.update_xaxes(dtick=2)
    return fig


def area_chart(current: pd.DataFrame):
    table = with_errors(share_table(current, "area_type"))
    table["name"] = [f"{a}<br>{s:.1f}% из {fmt_int(n)} ДТП"
                     for a, s, n in zip(table["area_type"], table["share"], table["n"])]
    order = [name for area in AREA_ORDER for name in table.loc[table["area_type"] == area, "name"]]
    fig = px.bar(table, x="name", y="share", color="area_type", error_y="plus", error_y_minus="minus",
                 color_discrete_map=AREA_COLORS, category_orders={"name": order},
                 hover_data={"n": True, "fatal": True, "plus": False, "minus": False, "name": False},
                 labels={"name": "", "share": SHARE, "area_type": "Территория",
                         "n": "Всего ДТП", "fatal": "С погибшими"})
    fig.update_layout(showlegend=False)
    return fig


def hour_chart(current: pd.DataFrame):
    table = with_errors(share_table(current, "hour"))
    overall = current["fatal"].mean() * 100
    fig = px.bar(table, x="hour", y="share", error_y="plus", error_y_minus="minus",
                 color_discrete_sequence=[BLUE],
                 hover_data={"n": True, "plus": False, "minus": False},
                 labels={"hour": "Час начала ДТП", "share": SHARE, "n": "Всего ДТП"})
    fig.add_hline(y=overall, line_width=1, line_color="#52514e",
                  annotation_text=f"все часы: {overall:.1f}%", annotation_position="top")
    fig.update_xaxes(dtick=2)
    return fig


def lighting_chart(current: pd.DataFrame, by_area: bool):
    data = current.dropna(subset=["light_group"])
    labels = {"light_group": "", "share": SHARE, "area_type": "Территория", "n": "Всего ДТП"}
    hover = {"n": True, "plus": False, "minus": False}
    if by_area:
        table = with_errors(share_table(data, ["light_group", "area_type"]))
        fig = px.bar(table, x="share", y="light_group", color="area_type", barmode="group",
                     orientation="h", error_x="plus", error_x_minus="minus",
                     color_discrete_map=AREA_COLORS, hover_data=hover, labels=labels,
                     category_orders={"light_group": LIGHT_ORDER, "area_type": AREA_ORDER})
        fig.update_layout(legend_traceorder="reversed")
        return fig
    table = with_errors(share_table(data, "light_group"))
    table["group"] = np.where(table["light_group"] == UNLIT, "Темно без освещения", "Остальные условия")
    table["light_group"] = [f"{g}<br>n = {fmt_int(n)}" for g, n in zip(table["light_group"], table["n"])]
    order = [name for group in LIGHT_ORDER for name in table["light_group"] if name.startswith(group + "<")]
    fig = px.bar(table, x="share", y="light_group", color="group", orientation="h",
                 error_x="plus", error_x_minus="minus",
                 color_discrete_map={"Темно без освещения": ACCENT, "Остальные условия": GRAY},
                 category_orders={"light_group": order}, hover_data=hover, labels=labels)
    fig.update_layout(showlegend=False)
    return fig


def season_charts(current: pd.DataFrame):
    counts = current.groupby("month").size().reindex(range(1, 13), fill_value=0)
    table = pd.DataFrame({"month": MONTHS, "n": counts.to_numpy()})
    left = px.bar(table, x="month", y="n", color_discrete_sequence=[BLUE],
                  labels={"month": "Месяц", "n": "Число ДТП за выбранный период"})
    shares = with_errors(share_table(current, "month"))
    shares["month"] = shares["month"].map(lambda m: MONTHS[int(m) - 1])
    right = px.scatter(shares, x="month", y="share", error_y="plus", error_y_minus="minus",
                       color_discrete_sequence=[ACCENT], category_orders={"month": MONTHS},
                       hover_data={"n": True, "plus": False, "minus": False},
                       labels={"month": "Месяц", "share": SHARE, "n": "Всего ДТП"})
    right.update_traces(marker_size=9)
    right.update_yaxes(rangemode="tozero")
    return left, right


def map_center(points: pd.DataFrame) -> dict:
    lat = points["latitude"].quantile([0.01, 0.99]).mean()
    lon = points["longitude"].quantile([0.01, 0.99]).mean()
    return {"lat": float(lat), "lon": float(lon)}


def points_map(points: pd.DataFrame, backdrop: bool):
    return px.scatter_map(
        points, lat="latitude", lon="longitude", color="severity",
        color_discrete_map=SEVERITY_COLORS, category_orders={"severity": SEVERITY_ORDER},
        hover_data=["id", "datetime", "category", "district"],
        labels={"severity": "Последствия", "category": "Тип ДТП", "district": "Район или город"},
        opacity=0.6, zoom=5.1, height=580, center=map_center(points),
        map_style="open-street-map" if backdrop else "white-bg",
    )


def grid_cells(points: pd.DataFrame, step: float) -> tuple[pd.DataFrame, dict]:
    cells = points.assign(
        cell_lat=np.floor(points["latitude"] / step) * step,
        cell_lon=np.floor(points["longitude"] / step) * step,
    ).groupby(["cell_lat", "cell_lon"]).agg(n=("id", "size"), fatal=("fatal", "sum")).reset_index()
    cells["share"] = cells["fatal"] / cells["n"] * 100
    cells["cell"] = cells.index.astype(str)
    features = [
        {"type": "Feature", "id": cell, "geometry": {"type": "Polygon", "coordinates": [[
            [lon, lat], [lon + step, lat], [lon + step, lat + step], [lon, lat + step], [lon, lat],
        ]]}}
        for cell, lat, lon in zip(cells["cell"], cells["cell_lat"], cells["cell_lon"])
    ]
    return cells, {"type": "FeatureCollection", "features": features}


def grid_map(points: pd.DataFrame, step: float, metric: str, min_count: int, backdrop: bool):
    cells, geojson = grid_cells(points, step)
    common = {
        "geojson": geojson, "locations": "cell", "opacity": 0.75, "zoom": 5.1, "height": 580,
        "center": map_center(points),
        "map_style": "open-street-map" if backdrop else "white-bg",
        "hover_data": {"n": True, "fatal": True, "share": ":.1f", "cell": False},
        "labels": {"n": "Всего ДТП", "fatal": "С погибшими", "share": SHARE},
    }
    if metric == "count":
        names = ["1-4", "5-19", "20-99", "100-499", "500 и больше"]
        cells["bin"] = pd.cut(cells["n"], [0, 4, 19, 99, 499, np.inf], labels=names).astype(str)
        fig = px.choropleth_map(cells, color="bin", color_discrete_map=dict(zip(names, COUNT_COLORS)),
                                category_orders={"bin": names}, **common)
        fig.update_layout(legend_title_text="ДТП в ячейке")
        return fig, cells
    cells = cells[cells["n"] >= min_count]
    fig = px.choropleth_map(cells, color="share", color_continuous_scale=SHARE_COLORS,
                            range_color=(0, 30), **common)
    fig.update_layout(coloraxis_colorbar_title_text="С погибшими, %")
    return fig, cells


def kind_of(category: pd.Series) -> pd.Series:
    kind = category.replace({"Съезд с дороги": "Съезд с дороги или опрокидывание",
                             "Опрокидывание": "Съезд с дороги или опрокидывание"})
    return kind.where(kind.isin(list(KIND_COLORS)), "Другое")


def projection_chart(data: pd.DataFrame, x: str, y: str, color: str, axis_names: tuple[str, str]):
    colors = {"area_type": AREA_COLORS, "severity": SEVERITY_COLORS, "kind": KIND_COLORS}[color]
    fig = px.scatter(
        data.assign(kind=kind_of(data["category"])), x=x, y=y, color=color,
        color_discrete_map=colors, category_orders={color: list(colors)}, opacity=0.55, height=460,
        hover_data={"category": True, "light_group": True, "area_type": True, "severity": True,
                    x: False, y: False},
        labels={x: axis_names[0], y: axis_names[1], "kind": "Тип ДТП", "category": "Тип ДТП",
                "light_group": "Освещение", "area_type": "Территория", "severity": "Последствия"},
    )
    fig.update_traces(marker_size=5)
    fig.update_layout(legend={"orientation": "h", "y": -0.15})
    return fig
