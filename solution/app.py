from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
from data_loading import AMUR_LAT, AMUR_LON, add_features, load_records, records_to_frame
from research import (
    LIT, PERPLEXITIES, PROJECTION_SIZE, SEED, UNLIT, YEARS, fmt_int, neighbor_agreement,
    run_projections, run_tests, share_table, tests_table,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ASSETS = Path(__file__).resolve().parent / "assets"
FILTER_PREFIX = "filter_"
FILTERS = {
    "area_type": ("Тип территории", "Все типы территорий"),
    "district": ("Район или город", "Все районы и города"),
    "category": ("Тип ДТП", "Все типы ДТП"),
}

st.set_page_config(page_title="Атлас аварийности", page_icon="🗺️", layout="wide")


@st.cache_data(show_spinner="Читаем выгрузку…")
def read_data(path: str, modified_ns: int, size: int) -> tuple[list[dict], pd.DataFrame]:
    records = load_records(Path(path))
    return records, add_features(records_to_frame(records))


@st.cache_data(show_spinner="Проверяем гипотезы…")
def cached_tests(path: str, modified_ns: int, size: int) -> list[dict]:
    return run_tests(read_data(path, modified_ns, size)[1])


@st.cache_data(show_spinner="Считаем PCA и t-SNE, это около 15 секунд…")
def cached_projections(path: str, modified_ns: int, size: int):
    return run_projections(read_data(path, modified_ns, size)[1])


def reset_filters() -> None:
    for key in list(st.session_state):
        if str(key).startswith(FILTER_PREFIX):
            del st.session_state[key]


def fatal_pct(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "нет ДТП"
    return f"{frame['fatal'].mean() * 100:.1f}% (n = {fmt_int(len(frame))})"


def show_overview(current: pd.DataFrame, start, end) -> None:
    st.subheader("1. Число ДТП по месяцам")
    st.plotly_chart(charts.monthly_chart(current, start, end), width="stretch")
    yearly = current[current["year"].between(*YEARS)].groupby("year").size()
    if len(yearly) > 1:
        st.caption(
            f"Полные годы в срезе: в {yearly.index[0]} г. - {fmt_int(yearly.iloc[0])} ДТП, "
            f"в {yearly.index[-1]} г. - {fmt_int(yearly.iloc[-1])}. Летом ДТП заметно больше, чем зимой. "
            "2026 г. представлен только январём, поэтому годы сравниваем по 2015-2025."
        )
    left, right = st.columns(2)
    with left:
        st.subheader("2. Типы ДТП")
        st.plotly_chart(charts.category_chart(current), width="stretch")
        st.caption("Подписи: число ДТП и % от всех ДТП в срезе. В подсказке - доля ДТП с погибшими. "
                   "Остальные 10 видов (каждый меньше 2% ДТП) объединены в «Прочие виды».")
    with right:
        st.subheader("3. Сколько человек пострадало в одном ДТП")
        st.plotly_chart(charts.victims_chart(current), width="stretch")
        one = (current["victims"] == 1).mean() * 100
        st.caption(f"В выгрузку попадают только ДТП с пострадавшими, поэтому минимум - 1 человек. "
                   f"В срезе {one:.0f}% ДТП с одним пострадавшим; максимум - "
                   f"{fmt_int(current['victims'].max())} чел.")
    st.subheader("4. Когда происходят ДТП: день недели и час")
    st.plotly_chart(charts.heatmap_chart(current), width="stretch")
    counts = current.groupby(["weekday", "hour"]).size()
    weekday, hour = counts.idxmax()
    st.caption(f"Цвет - число ДТП в ячейке за выбранный период. Больше всего ДТП в срезе: "
               f"{charts.WEEKDAYS[int(weekday)]}, {int(hour)} ч - {counts.max()} ДТП. На всех данных "
               "пик - вечерний час пик в будни (17-18 ч), а ночью ДТП больше в ночи на субботу и воскресенье.")


def show_severity(current: pd.DataFrame) -> None:
    st.markdown(
        "Во всех графиках раздела показатель один: **доля ДТП с погибшими** - ДТП (>=1 человек)"
        ", в % от всех ДТП группы. Отрезки - 95% доверительный интервал Уилсона."
    )
    st.subheader("5. Благовещенск, другие города и районы")
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.area_chart(current), width="stretch")
    table = share_table(current, "district").sort_values("share", ascending=False)
    right.dataframe(
        table[["district", "n", "fatal", "share", "low", "high"]].round(1), hide_index=True,
        height=400, width="stretch",
        column_config={"district": "Район или город", "n": "ДТП", "fatal": "С погибшими",
                       "share": "Доля, %", "low": "ДИ от", "high": "ДИ до"},
    )
    st.caption("'Други города': Белогорск, Свободный, Тында, Зея, Шимановск (в выгрузке у них "
               "территория «Амурская область», в таблице - «Города обл. значения»), Райчихинск и "
               "Углегорск. «Районы» - муниципальные районы: сёла, посёлки и трассы между ними.")
    st.subheader("6. В какое время суток ДТП смертельнее?")
    st.plotly_chart(charts.hour_chart(current), width="stretch")
    night = current["hour"].isin([21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
    st.caption(f"В срезе с 21 до 6 ч доля ДТП с погибшими {fatal_pct(current[night])}, с 7 до 20 ч - "
               f"{fatal_pct(current[~night])}. На всех данных ночью она почти вдвое выше. Отсюда "
               "следующий вопрос: дело в темноте как таковой или в том, что дорога не освещена?")
    st.subheader("7. Вопрос из графика 6: темнота или отсутствие освещения?")
    by_area = st.checkbox("Разбить по типу территории", key="lighting_by_area")
    st.plotly_chart(charts.lighting_chart(current, by_area), width="stretch")
    light = current["light_group"]
    st.caption("На всех данных освещённая темнота похожа на день. "
               "Разбивка по территориям уточняет: освещённые улицы в основном в городах, где ДТП и днём "
               "легче, поэтому часть сходства с днём - эффект состава. В городах темнота и при фонарях "
               "опаснее дня, но темнота без освещения - самое опасное условие в каждом типе территории. ")
    st.subheader("8. Зимой ДТП меньше, но тяжелее ли они?")
    left, right = st.columns(2)
    count_fig, share_fig = charts.season_charts(current)
    left.plotly_chart(count_fig, width="stretch")
    right.plotly_chart(share_fig, width="stretch")
    season = current["season"]
    st.caption(f"В срезе зимой (дек-фев) доля ДТП с погибшими "
               f"{fatal_pct(current[season == 'Зима'])}, летом (июн-авг) - {fatal_pct(current[season == 'Лето'])}. "
               "Число ДТП летом выше, а доля с погибшими по месяцам меняется мало. Проверка - гипотеза H3.")


def show_map(current: pd.DataFrame) -> None:
    missing = int(current["latitude"].isna().sum())
    zeros = int(((current["latitude"] == 0) & (current["longitude"] == 0)).sum())
    outside = int((current["map_valid"] & ~current["in_region"]).sum())
    swapped = int((current["longitude"].between(*AMUR_LAT) & current["latitude"].between(*AMUR_LON)).sum())
    points = current[current["in_region"]].copy()
    st.caption(
        f"На карте {fmt_int(len(points))} из {fmt_int(len(current))} ДТП среза. Не показаны: "
        f"без координат - {missing}, координаты (0, 0) - {zeros}, точка вне границ Амурской области "
        f"(широта {AMUR_LAT[0]}-{AMUR_LAT[1]}°, долгота {AMUR_LON[0]}-{AMUR_LON[1]}°) - {outside}. "
        f"Точек, у которых широта и долгота перепутаны местами: {swapped}."
    )
    if points.empty:
        st.info("В срезе нет пригодных для карты координат.")
        return
    view = st.radio("Представление", ["Точки", "Сетка: число ДТП", "Сетка: доля ДТП с погибшими"],
                    horizontal=True, key="map_view")
    backdrop = st.checkbox("Показывать подложку карты", value=True)
    if view == "Точки":
        if len(points) > 5000:
            points = points.sample(5000, random_state=SEED)
        st.plotly_chart(charts.points_map(points, backdrop), width="stretch")
    else:
        left, right = st.columns(2)
        step = left.select_slider("Размер ячейки, градусы", [0.1, 0.25, 0.5], value=0.25)
        min_count = right.slider("Минимум ДТП в ячейке для доли", 5, 50, 20, step=5,
                                 disabled=view == "Сетка: число ДТП")
        metric = "count" if view == "Сетка: число ДТП" else "share"
        fig, cells = charts.grid_map(points, step, metric, min_count, backdrop)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Ячейка {step}° × {step}° - это около {step * 111:.0f} км с севера на юг и "
                   f"{step * 70:.0f} км с запада на восток. Ячеек на карте: {len(cells)}."
                   + (f" Показаны ячейки, где не меньше {min_count} ДТП." if metric == "share" else ""))
    st.caption(charts.MAP_NOTE + " Координаты взяты из выгрузки.")
    st.markdown(
        "**Что видно** (все ДТП, ячейка 0.25°). Больше всего ДТП в ячейке Благовещенска (около 3 800), "
        "но с погибшими там только 3%. Ячейки с долей 20-30% лежат вне городов: вдоль трассы "
        "Р-297 «Амур» (Сковородино - Магдагачи - Шимановск - Свободный) и на дорогах между "
        "райцентрами. Это та же закономерность, что на графиках 5 и 7: за городом больше "
        "неосвещённых перегонов и, видимо, выше скорости."
    )


def show_projections(path: str, stat) -> None:
    data, loadings, ratio = cached_projections(path, stat.st_mtime_ns, stat.st_size)
    color = st.radio("Цвет точек", ["area_type", "kind", "severity"], horizontal=True,
                     format_func={"area_type": "Тип территории", "kind": "Тип ДТП",
                                  "severity": "Последствия"}.get, key="projection_color")
    st.markdown(
        "Признаки - условия ДТП: тип ДТП (8 групп), освещение (4), тип территории (3), место "
        "(перегон, перекрёсток, пешеходный переход), осадки, зима, выходной, есть ли мотоцикл или "
        "грузовик, число ТС и участников. "
    )
    st.subheader("11. PCA")
    left, right = st.columns([3, 2])
    left.plotly_chart(charts.projection_chart(
        data, "pc1", "pc2", color, (f"PC1 ({ratio[0]:.1%} дисперсии)", f"PC2 ({ratio[1]:.1%} дисперсии)"),
    ), width="stretch")
    top = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index).head(10)
    right.dataframe(top.round(2), width="stretch")
    right.caption(f"10 признаков с наибольшим по модулю весом в PC1. Две компоненты вместе сохраняют "
                  f"{ratio.sum():.1%} дисперсии.")
    st.markdown(
        "**PC1 - ось «город ↔ трасса».** С одной стороны перекрёстки, Благовещенск, столкновения и "
        "пешеходные переходы, с другой - районы, перегоны, темнота без освещения, съезды с дороги и "
        "опрокидывания. ДТП с погибшими чаще лежат на «трассовой» стороне."
    )
    st.subheader("12. t-SNE")
    columns = st.columns(len(PERPLEXITIES))
    for column, perplexity in zip(columns, PERPLEXITIES):
        column.plotly_chart(charts.projection_chart(
            data, f"tsne{perplexity}_x", f"tsne{perplexity}_y", color, ("t-SNE 1", "t-SNE 2"),
        ), width="stretch")
        column.caption(f"perplexity = {perplexity}, init = pca, random_state = {SEED}")
    st.markdown(
        "**Сравнение.** При perplexity 30 ДТП собираются в несколько десятков плотных островков. "
        "Каждый островок - ДТП с почти одинаковым набором условий: один тип ДТП, одна территория, "
        "одно освещение. Островки одного типа лежат рядом (столкновения, наезды на пешеходов, "
        "съезды и опрокидывания). При perplexity 5 островки не отделяются друг от друга, получается "
        "одно облако, но те же области по типу ДТП и территории в нём видны. Значит, устойчиво "
        "разделение по типу ДТП и территории, а число и форма «кластеров» зависят от параметра. "
        "Размеры пятен и расстояния между ними не интерпретируем."
    )
    agreement = neighbor_agreement(data)
    st.dataframe(agreement.round(0).astype(int), width="stretch")
    st.caption("Проверка по исходным признакам: сколько % из 10 ближайших соседей точки на проекции "
               "совпадают с ней по признаку. Последняя строка - сколько совпадало бы у случайных соседей.")
    st.markdown(
        "Соседи на обеих картах t-SNE почти всегда того же типа ДТП, с той же территории и тем же "
        "освещением. По наличию погибших совпадение на уровне случайного: ДТП с погибшими не "
        "образуют отдельной группы. Условия повышают вероятность гибели, но не определяют исход."
    )


def show_tests(path: str, stat) -> None:
    results = cached_tests(path, stat.st_mtime_ns, stat.st_size)
    st.table(tests_table(results).set_index("Гипотеза").T)
    st.markdown(
        "Единица анализа - одно ДТП, каждое ДТП входит в группу один раз. Уровень значимости 0.05; "
        "с поправкой Бонферрони на три проверки он равен 0.017, выводы от этого не меняются.\n\n"
        "- **H1.** H0: в тёмное время доля ДТП с погибшими одинакова на освещённых и неосвещённых "
        "участках. хи квадрат-тест подходит: две категории на две\n"
        "- **H2.** H0: метка «районы / Благовещенск» не связана с исходом ДТП. Тогда метки можно "
        "перемешивать, поэтому применён перестановочный тест, а ДИ разности посчитан бутстрэпом ДТП "
        "внутри каждой группы.\n"
        "- **H3.** H0: среднее число погибших на 100 ДТП зимой и летом одинаково. Признак сильно "
        "скошен (обычно 0), но групп в тысячи ДТП хватает, чтобы среднее было близко к нормальному, "
        "поэтому применён t-тест Уэлча. Разница близка к нулю, и ДИ её ограничивает: примерно "
        "+-2 погибших на 100 ДТП.\n\n"
        "Гипотеза H1 возникла после просмотра графика 6, поэтому её p-value немного оптимистичен. "
        "Эффект при этом очень большой."
    )


def show_before_after() -> None:
    left, right = st.columns(2)
    left.markdown("**До**")
    left.image(str(ASSETS / "lighting_before.png"))
    right.markdown("**После**")
    right.image(str(ASSETS / "lighting_after.png"))
    st.markdown(
        "Вопрос и метрика не менялись: доля ДТП с погибшими при разном освещении. Что изменилось:\n"
        "- длинные повёрнутые подписи заменены короткими горизонтальными, рядом - размер группы;\n"
        "- группы идут по порядку от дня к полной темноте, а не по алфавиту;\n"
        "- доли переведены в проценты с подписью оси, у каждого столбца 95% ДИ;\n"
        "- убран столбец «Не установлено»: 3 ДТП давали самый высокий столбец (33%) и сбивали с толку;\n"
        "- серый цвет для всех групп и акцент на одной, заголовок называет вывод.\n\n"
        "Теперь сразу видно главное: в темноте без освещения доля ДТП с погибшими 20.0%, в 2.7 раза "
        "выше, чем в темноте при освещении (7.5%), и выше, чем в любой другой группе."
    )


def show_conclusions() -> None:
    st.markdown(
        """
**Три главных вывода** (все ДТП 2015-2025, n = 12 990):

1. **Большинство ДТП происходит в городах, большинство смертей - вне их.** В Благовещенске
   34% всех ДТП, но с погибшими только 3.0% из них. В районах 14.9%: разница +11.9 п.п.,
   95% ДИ [10.8; 12.9], перестановочный тест p < 0.001. На районы приходится 74% всех ДТП
   с погибшими (графики 5, карта, H2).
2. **Самое опасное условие - темнота без освещения.** Ночью доля ДТП с погибшими примерно вдвое
   выше, чем днём. В темноте без освещения она 20.0%, при включённом освещении - 7.5%: +12.6 п.п.,
   95% ДИ [10.5; 14.6], хи квадрат p < 0.001. Разрыв есть в каждом типе территории: Благовещенск
   10.2% и 4.8%, другие города 20.1% и 8.7%, районы 20.8% и 12.6% (графики 6-7, H1).
3. **Зима уменьшает число ДТП, но не их тяжесть.** С декабря по февраль ДТП в месяц на 42%
   меньше, чем с июня по август (72 и 125), а погибших на 100 ДТП столько же: 10.0 и 10.2,
   разница -0.2, 95% ДИ [-2.0; +1.6], p = 0.86 (график 8, H3).

Это наблюдения. Почему так происходит (скорость на трассах, отсутствие фонарей и тротуаров,
меньше поездок зимой), по выгрузке проверить нельзя. Для оценки риска нужен транспортный поток.

**Что изучить дальше.** Наезды на пешеходов в районах в темноте без освещения: примерно каждый
третий такой наезд заканчивается гибелью (32.5%, n = 338). Стоит найти на карте участки, где такие
ДТП повторяются, например сёла вдоль Р-297, и сравнить их с данными об освещении и потоке.
"""
    )


def show_records(current: pd.DataFrame, records: list[dict]) -> None:
    columns = ["source_row", "id", "datetime", "district", "area_type", "category", "severity",
               "light_group", "participants_count", "injured_count", "dead_count",
               "latitude", "longitude"]
    st.dataframe(current[columns].head(500), hide_index=True, width="stretch")
    st.caption("Первые 500 записей текущего среза; CSV содержит весь срез.")
    st.download_button(
        "Скачать текущий срез CSV", current[columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="selected_records.csv", mime="text/csv",
    )
    lookup = current.set_index("source_row")
    chosen = st.selectbox(
        "Открыть исходную запись", current["source_row"].tolist(),
        format_func=lambda index: f"Строка {index} · ID {lookup.at[index, 'id']}",
    )
    st.json(records[int(chosen)], expanded=False)


def main() -> None:
    st.title("Атлас аварийности: Амурская область")
    st.caption("ДЗ 2 · данные «Карты ДТП» (dtp-stat.ru) · единица анализа - одно ДТП")
    files = sorted(DATA_DIR.glob("*.geojson")) + sorted(DATA_DIR.glob("*.json"))
    files = [p for p in files if not p.name.endswith(".source.json")]
    if not files:
        st.warning("В data/raw пока нет выгрузки. Скачайте файл своего региона.")
        st.code(
            "python data/assign_region.py YOUR_GITHUB_USERNAME\n"
            "python data/import_data.py /path/to/region.geojson.zip",
            language="bash",
        )
        st.link_button("Открыть данные «Карты ДТП»", "https://dtp-stat.ru/opendata/")
        st.markdown("Инструкция находится в `data/README.md` репозитория.")
        return

    selected_path = st.sidebar.selectbox(
        "Выгрузка", files, format_func=lambda p: p.name,
        key="source_file", on_change=reset_filters,
    )
    try:
        stat = selected_path.stat()
        records, frame = read_data(str(selected_path), stat.st_mtime_ns, stat.st_size)
    except (ValueError, OSError, TypeError) as error:
        st.error(f"Не удалось прочитать выгрузку: {error}")
        return
    if frame.empty:
        st.warning("Файл содержит пустой список записей.")
        return

    invalid_dates = int(frame["datetime"].isna().sum())
    duplicate_ids = int(frame.loc[frame["id"].notna(), "id"].duplicated().sum())
    st.sidebar.caption(f"Исходных записей: {len(frame):,}")
    st.sidebar.caption(
        f"Без распознанной даты: {invalid_dates:,}. "
        f"Повторных непустых ID: {duplicate_ids:,}. Дубликаты не удалены."
    )
    dated = frame[frame["datetime"].notna()]
    if dated.empty:
        st.error("Нет распознанных дат. Проверьте исходный формат и загрузчик.")
        st.dataframe(frame.head(20), hide_index=True)
        return

    first, last = dated["datetime"].min().date(), dated["datetime"].max().date()
    default = (max(first, date(YEARS[0], 1, 1)), min(last, date(YEARS[1], 12, 31)))
    if default[0] > default[1]:
        default = (first, last)
    st.sidebar.button("Сбросить фильтры", on_click=reset_filters)
    period = st.sidebar.date_input(
        "Период", value=default, min_value=first, max_value=last,
        key="filter_period",
    )
    st.sidebar.caption(f"По умолчанию выбраны полные годы {YEARS[0]}-{YEARS[1]}. "
                       f"Данные есть с {first:%d.%m.%Y} по {last:%d.%m.%Y}.")
    if len(period) != 2:
        st.info("Выберите начало и конец периода.")
        return
    start, end = period
    current = dated[
        (dated["datetime"] >= pd.Timestamp(start))
        & (dated["datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    ].copy()
    for column, (title, all_label) in FILTERS.items():
        include_all = st.sidebar.checkbox(all_label, value=True, key=f"filter_all_{column}")
        if include_all:
            continue
        options = sorted(frame[column].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(title, options, default=[], key=f"filter_{column}")
        current = current[current[column].astype("string").isin(selected)].copy()

    st.caption(
        f"Период: {start:%d.%m.%Y} - {end:%d.%m.%Y}. Графики разделов «Обзор», «Тяжесть» и «Карта» "
        "и показатели ниже построены по выбранному срезу."
    )
    if invalid_dates:
        st.warning(f"Из фильтра по времени исключено записей без даты: {invalid_dates:,}.")
    if current.empty:
        st.warning("В выбранном срезе нет записей. Измените или сбросьте фильтры.")
        return
    a, b, c, d, e = st.columns(5)
    a.metric("ДТП в срезе", fmt_int(len(current)))
    b.metric("ДТП с погибшими", fmt_int(current["fatal"].sum()))
    c.metric("Доля ДТП с погибшими", f"{current['fatal'].mean():.1%}")
    d.metric("Погибло, чел.", fmt_int(current["dead_count"].sum()))
    e.metric("Ранено, чел.", fmt_int(current["injured_count"].sum()))
    if len(current) < 30:
        st.warning(
            "В выборке меньше 30 записей. Посмотрите, как отдельные ДТП влияют на результат. "
            "Число 30 здесь выбрано для напоминания, а не как критерий надёжности."
        )

    tabs = st.tabs(["Обзор", "Тяжесть", "Карта", "PCA и t-SNE", "Гипотезы", "До и после",
                    "Выводы", "Исходные записи"])
    with tabs[0]:
        show_overview(current, start, end)
    with tabs[1]:
        show_severity(current)
    with tabs[2]:
        show_map(current)
    with tabs[3]:
        show_projections(str(selected_path), stat)
    with tabs[4]:
        show_tests(str(selected_path), stat)
    with tabs[5]:
        show_before_after()
    with tabs[6]:
        show_conclusions()
    with tabs[7]:
        show_records(current, records)


if __name__ == "__main__":
    main()
