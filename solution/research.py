"""Заранее считаемые результаты: три гипотезы, PCA/t-SNE и график «до/после».

Запуск из корня репозитория: .venv/bin/python solution/research.py
Печатает числа для отчёта и сохраняет картинки «до/после» в solution/assets/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

from data_loading import add_features, load_records, records_to_frame

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
SEED = 42
YEARS = (2015, 2025)
ALPHA = 0.05
RESAMPLES = 9999
PROJECTION_SIZE = 3000
PERPLEXITIES = (30, 5)

LIT, UNLIT = "Темно, освещение есть", "Темно, освещения нет"
LIGHT_ORDER = ["День", "Сумерки", LIT, UNLIT]
AREA_ORDER = ["Благовещенск", "Другие города", "Районы"]
TOP_CATEGORIES = [
    "Столкновение", "Наезд на пешехода", "Съезд с дороги", "Опрокидывание",
    "Наезд на препятствие", "Наезд на велосипедиста", "Наезд на стоящее ТС",
]
FLAG_NAMES = {
    "stretch": "Перегон (нет объектов рядом)",
    "intersection": "Перекрёсток",
    "crossing": "Пешеходный переход",
    "bad_weather": "Осадки или туман",
    "has_moto": "Есть мотоцикл или мопед",
    "has_truck": "Есть грузовик",
}


def full_years(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["year"].between(*YEARS)]


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def share_table(frame: pd.DataFrame, by) -> pd.DataFrame:
    """Доля ДТП с погибшими в каждой группе, % и 95% ДИ Уилсона."""
    table = frame.groupby(by, observed=True)["fatal"].agg(n="size", fatal="sum").reset_index()
    low, high = proportion_confint(table["fatal"], table["n"], alpha=ALPHA, method="wilson")
    table["share"] = table["fatal"] / table["n"] * 100
    table["low"] = np.asarray(low) * 100
    table["high"] = np.asarray(high) * 100
    return table


def mean_diff(x, y, axis=-1):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def test_lighting(frame: pd.DataFrame) -> dict:
    """H1: в темноте без освещения ДТП чаще заканчиваются гибелью, чем при освещении."""
    counts = frame.groupby("light_group")["fatal"].agg(["sum", "size"])
    (k1, n1), (k2, n2) = counts.loc[UNLIT], counts.loc[LIT]
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    chi2, p_value, _, expected = stats.chi2_contingency(table, correction=False)
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "name": "H1. Освещение в тёмное время",
        "groups": (UNLIT, LIT), "n": (n1, n2), "values": (p1 * 100, p2 * 100),
        "metric": "доля ДТП с погибшими, %",
        "effect": (p1 - p2) * 100, "ci": ((p1 - p2 - 1.96 * se) * 100, (p1 - p2 + 1.96 * se) * 100),
        "ci_method": "нормальное приближение (Вальд)",
        "test": "χ² Пирсона для таблицы 2×2", "statistic": chi2, "p_value": p_value,
        "ratio": p1 / p2, "min_expected": expected.min(),
    }


def test_districts(frame: pd.DataFrame) -> dict:
    """H2: в районах ДТП чаще заканчиваются гибелью, чем в Благовещенске."""
    districts = frame.loc[frame["area_type"] == "Районы", "fatal"].to_numpy(float)
    city = frame.loc[frame["area_type"] == "Благовещенск", "fatal"].to_numpy(float)
    perm = stats.permutation_test(
        (districts, city), mean_diff, vectorized=True, n_resamples=RESAMPLES, batch=500, rng=SEED,
    )
    boot = stats.bootstrap(
        (districts, city), mean_diff, vectorized=True, n_resamples=RESAMPLES, batch=500,
        method="percentile", rng=SEED,
    )
    return {
        "name": "H2. Районы и Благовещенск",
        "groups": ("Районы", "Благовещенск"), "n": (len(districts), len(city)),
        "values": (districts.mean() * 100, city.mean() * 100),
        "metric": "доля ДТП с погибшими, %",
        "effect": perm.statistic * 100,
        "ci": (boot.confidence_interval.low * 100, boot.confidence_interval.high * 100),
        "ci_method": f"bootstrap, перцентильный, {RESAMPLES} повторов",
        "test": f"перестановочный тест, {RESAMPLES} перестановок меток", "statistic": perm.statistic,
        "p_value": perm.pvalue, "ratio": districts.mean() / city.mean(),
    }


def test_winter(frame: pd.DataFrame) -> dict:
    """H3: зимой ДТП тяжелее, чем летом (погибших на 100 ДТП)."""
    winter = frame.loc[frame["season"] == "Зима", "dead_count"].to_numpy(float) * 100
    summer = frame.loc[frame["season"] == "Лето", "dead_count"].to_numpy(float) * 100
    result = stats.ttest_ind(winter, summer, equal_var=False)
    ci = result.confidence_interval(1 - ALPHA)
    return {
        "name": "H3. Зима и лето",
        "groups": ("Зима (дек-фев)", "Лето (июн-авг)"), "n": (len(winter), len(summer)),
        "values": (winter.mean(), summer.mean()),
        "metric": "погибших на 100 ДТП",
        "effect": winter.mean() - summer.mean(), "ci": (ci.low, ci.high),
        "ci_method": "t-распределение Уэлча",
        "test": "t-тест Уэлча", "statistic": result.statistic, "p_value": result.pvalue,
        "ratio": winter.mean() / summer.mean(),
    }


def run_tests(frame: pd.DataFrame) -> list[dict]:
    data = full_years(frame)
    return [test_lighting(data), test_districts(data), test_winter(data)]


def format_p(p_value: float) -> str:
    return "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"


def tests_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        unit = "п.п." if "доля" in r["metric"] else ""
        rows.append({
            "Гипотеза": r["name"],
            "Группы (n)": f"{r['groups'][0]} ({fmt_int(r['n'][0])}) и {r['groups'][1]} ({fmt_int(r['n'][1])})",
            "Показатель": f"{r['metric']}: {r['values'][0]:.1f} и {r['values'][1]:.1f}",
            "Эффект": f"{r['effect']:+.1f} {unit}".strip(),
            "95% ДИ эффекта": f"[{r['ci'][0]:+.1f}; {r['ci'][1]:+.1f}], {r['ci_method']}",
            "Тест": r["test"],
            "p-value": format_p(r["p_value"]),
            "Вывод при α = 0.05": "H0 отвергаем" if r["p_value"] < ALPHA else "H0 не отвергаем",
        })
    return pd.DataFrame(rows)


def projection_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Подвыборка ДТП и признаки условий. Исход (погибшие, раненые) в признаки не входит."""
    data = full_years(frame).dropna(subset=["light_group"])
    data = data.sample(min(PROJECTION_SIZE, len(data)), random_state=SEED)
    category = data["category"].where(data["category"].isin(TOP_CATEGORIES), "Прочие")
    features = pd.concat([
        pd.get_dummies(category, prefix="Тип"),
        pd.get_dummies(data["light_group"], prefix="Свет"),
        pd.get_dummies(data["area_type"], prefix="Территория"),
        data[list(FLAG_NAMES)].rename(columns=FLAG_NAMES),
        (data["season"] == "Зима").rename("Зима"),
        (data["weekday"] >= 5).rename("Выходной"),
        data[["vehicles_count", "participants_count"]].rename(
            columns={"vehicles_count": "Число ТС", "participants_count": "Число участников"}
        ),
    ], axis=1).astype(float)
    return data, features


def run_projections(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data, features = projection_features(frame)
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=SEED).fit(scaled)
    data = data.copy()
    data[["pc1", "pc2"]] = pca.transform(scaled)
    for perplexity in PERPLEXITIES:
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=SEED)
        data[[f"tsne{perplexity}_x", f"tsne{perplexity}_y"]] = tsne.fit_transform(scaled)
    loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=["PC1", "PC2"])
    return data, loadings, pca.explained_variance_ratio_


def neighbor_agreement(data: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Какая доля из k ближайших соседей точки на проекции совпадает с ней по исходному признаку."""
    labels = {"Тип ДТП": "category", "Территория": "area_type", "Освещение": "light_group",
              "Есть погибшие": "fatal"}
    views = {"PCA": ("pc1", "pc2")}
    views.update({f"t-SNE, perplexity {p}": (f"tsne{p}_x", f"tsne{p}_y") for p in PERPLEXITIES})
    rows = {}
    for view, columns in views.items():
        points = data[list(columns)].to_numpy()
        nearest = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points, return_distance=False)
        rows[view] = {
            name: (data[column].to_numpy()[nearest[:, 1:]] == data[column].to_numpy()[:, None]).mean() * 100
            for name, column in labels.items()
        }
    rows["Случайные соседи"] = {
        name: (data[column].value_counts(normalize=True) ** 2).sum() * 100 for name, column in labels.items()
    }
    return pd.DataFrame(rows).T


def lighting_before(frame: pd.DataFrame):
    """Первая версия: как получилось «из коробки»."""
    fig, ax = plt.subplots(figsize=(8, 6))
    frame.groupby("light")["fatal"].mean().plot.bar(ax=ax)
    fig.tight_layout()
    return fig


def lighting_after(frame: pd.DataFrame):
    table = share_table(frame.dropna(subset=["light_group"]), "light_group")
    table = table.set_index("light_group").loc[LIGHT_ORDER]
    overall = frame["fatal"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(table))
    colors = ["#b4b2a9", "#b4b2a9", "#b4b2a9", "#B33440"]
    errors = [table["share"] - table["low"], table["high"] - table["share"]]
    ax.barh(y, table["share"], xerr=errors, color=colors, height=0.6,
            error_kw={"ecolor": "#52514e", "capsize": 3, "lw": 1})
    ax.set_yticks(y, [f"{name}\nn = {fmt_int(n)}" for name, n in zip(table.index, table["n"])])
    ax.invert_yaxis()
    for i, (share, high) in enumerate(zip(table["share"], table["high"])):
        ax.text(high + 0.4, i, f"{share:.1f}%", va="center", color="#0b0b0b")
    ax.set_xlim(0, table["high"].max() + 4)
    ax.set_xlabel("ДТП с погибшими, % от всех ДТП при этом освещении")
    ax.set_title("В темноте смертельных ДТП больше там, где нет освещения", loc="left", fontsize=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e1e0d9", lw=0.8)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, f"Амурская область, ДТП {YEARS[0]}-{YEARS[1]}; в среднем по всем ДТП {overall:.1f}%. "
             "Отрезки - 95% ДИ Уилсона. Источник: dtp-stat.ru", size=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> None:
    files = sorted((ROOT / "data" / "raw").glob("*.geojson"))
    if not files:
        raise SystemExit("В data/raw нет выгрузки.")
    frame = add_features(records_to_frame(load_records(files[0])))
    data = full_years(frame)
    print(f"Файл: {files[0].name}; ДТП за {YEARS[0]}-{YEARS[1]}: {len(data):,}")
    for r in run_tests(frame):
        print(f"\n{r['name']}: {r['test']}")
        for key in ("groups", "n", "values", "effect", "ci", "statistic", "p_value", "ratio"):
            print(f"  {key}: {r[key]}")
        if "min_expected" in r:
            print(f"  min_expected: {r['min_expected']:.1f}")
    ASSETS.mkdir(exist_ok=True)
    for name, draw in (("lighting_before.png", lighting_before), ("lighting_after.png", lighting_after)):
        fig = draw(data)
        fig.savefig(ASSETS / name, dpi=150)
        plt.close(fig)
    print(f"\nКартинки «до/после» сохранены в {ASSETS}")


if __name__ == "__main__":
    main()
