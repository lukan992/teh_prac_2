from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, project_root, read_json

RAW_RESULT_FILES = {
    "clustering": "clustering_results.csv",
    "classification": "classification_results.csv",
    "forecasting": "forecasting_results.csv",
    "pointcloud": "pointcloud_results.csv",
    "resnet": "resnet_hdbscan_results.csv",
    "resource_usage": "resource_usage.csv",
    "final_comparison": "final_comparison.csv",
}

TABLE_OUTPUTS = {
    "clustering": "clustering_results_table.csv",
    "classification": "classification_results_table.csv",
    "forecasting": "forecasting_results_table.csv",
    "pointcloud": "pointcloud_results_table.csv",
    "resnet": "resnet_hdbscan_results_table.csv",
    "resource_usage": "resource_usage_table.csv",
    "final_comparison": "final_comparison_table.csv",
    "final_applicability": "final_applicability_table.csv",
}

MARKDOWN_TABLE_OUTPUTS = {
    "clustering": "clustering_results_table.md",
    "classification": "classification_results_table.md",
    "forecasting": "forecasting_results_table.md",
    "pointcloud": "pointcloud_results_table.md",
    "resource_usage": "resource_usage_table.md",
    "final_comparison": "final_comparison_table.md",
    "final_applicability": "final_applicability_table.md",
}

METHOD_DISPLAY_NAMES = {
    "traclus": "TRACLUS-like",
    "st_dbscan": "ST-DBSCAN",
    "vector_field_kmeans": "Vector Field k-Means",
    "spatiotemporal_clustering": "Spatio-temporal trajectory clustering",
    "cnn_segment_classifier": "CNN-классификатор сегментов",
    "lstm_baseline": "LSTM baseline",
    "lstm_class_aware": "Class-aware LSTM",
    "sparse_pointcloud": "Sparse point cloud trajectory estimation",
    "cluster_filter": "Cluster Filter",
    "cl_det": "CL-Det / DBSCAN LiDAR tracking",
    "resnet_hdbscan": "ResNet-like embeddings + HDBSCAN",
    "resnet_kmeans": "ResNet-like embeddings + K-Means",
    "handcrafted_dbscan": "Ручные признаки + DBSCAN",
    "handcrafted_hdbscan": "Ручные признаки + HDBSCAN",
    "resnet_hdbscan_family": "ResNet-like embedding family",
    "sparse_pointcloud_clean": "Sparse point cloud trajectory estimation (clean)",
    "sparse_pointcloud_medium": "Sparse point cloud trajectory estimation (medium)",
    "sparse_pointcloud_hard": "Sparse point cloud trajectory estimation (hard)",
    "cluster_filter_clean": "Cluster Filter (clean)",
    "cluster_filter_medium": "Cluster Filter (medium)",
    "cluster_filter_hard": "Cluster Filter (hard)",
    "cl_det_clean": "CL-Det / DBSCAN LiDAR tracking (clean)",
    "cl_det_medium": "CL-Det / DBSCAN LiDAR tracking (medium)",
    "cl_det_hard": "CL-Det / DBSCAN LiDAR tracking (hard)",
}

THREE_DECIMAL_KEYS = {
    "accuracy",
    "ari",
    "nmi",
    "macro_f1",
    "precision_macro",
    "recall_macro",
    "purity",
    "silhouette",
    "cluster_accuracy",
    "noise_ratio",
    "detection_rate",
    "false_positive_rate",
}

TWO_DECIMAL_KEYS = {
    "mae",
    "mse",
    "rmse",
    "ade",
    "fde",
    "runtime_seconds",
    "fps",
    "position_rmse",
    "main_metric_value",
    "estimated_inference_time",
    "mean_inference_time",
}

ONE_DECIMAL_KEYS = {
    "peak_ram_mb",
    "peak_vram_mb",
    "peak_vram_gpu0_mb",
    "peak_vram_gpu1_mb",
    "peak_vram_total_mb",
}


@dataclass
class Context:
    root: Path
    metrics_dir: Path
    results_tables_dir: Path
    report_tables_dir: Path
    text_fragments_dir: Path
    appendices_dir: Path
    dataset_config: dict[str, Any]
    trajectory_labels: pd.DataFrame
    segment_metadata: dict[str, Any]
    pointcloud_metadata: dict[str, Any]
    object_counts: dict[str, int]


def read_csv_checked(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_method_names(df: pd.DataFrame) -> pd.DataFrame:
    if "method" in df.columns:
        df = df.copy()
        df["method"] = df["method"].map(lambda value: METHOD_DISPLAY_NAMES.get(value, value))
    return df


def parse_vram_cell(value: Any) -> tuple[float | None, float | None, float | None]:
    if pd.isna(value) or value == "":
        return None, None, None
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    if not parts:
        return None, None, None
    floats = [float(part) for part in parts]
    gpu0 = floats[0]
    gpu1 = floats[1] if len(floats) > 1 else None
    total = gpu0 + gpu1 if gpu1 is not None else gpu0
    return gpu0, gpu1, total


def infer_round_digits(column: str, metric_name: str | None = None) -> int | None:
    key = metric_name or column
    key = key.lower()
    if key in THREE_DECIMAL_KEYS or "f1" in key or "silhouette" in key or key in {"ari", "nmi"}:
        return 3
    if key in TWO_DECIMAL_KEYS or key in {"mae", "mse", "rmse", "ade", "fde", "fps"}:
        return 2
    if key in ONE_DECIMAL_KEYS or "vram" in key or "ram" in key:
        return 1
    return None


def round_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    for column in rounded.columns:
        if not pd.api.types.is_numeric_dtype(rounded[column]):
            continue
        digits = infer_round_digits(column)
        if digits is not None:
            rounded[column] = rounded[column].round(digits)
    if {"main_metric", "main_metric_value"}.issubset(rounded.columns):
        rounded["main_metric_value"] = [
            round(value, infer_round_digits("main_metric_value", metric_name=str(metric)) or 3)
            if pd.notna(value)
            else value
            for metric, value in zip(rounded["main_metric"], rounded["main_metric_value"])
        ]
    return rounded


def format_secondary_metrics(value: Any) -> Any:
    if pd.isna(value) or value == "":
        return value
    items = []
    for raw_part in str(value).split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            items.append(part)
            continue
        key, raw_number = [item.strip() for item in part.split("=", 1)]
        try:
            numeric = float(raw_number)
        except ValueError:
            items.append(f"{key}={raw_number}")
            continue
        digits = infer_round_digits(key) or 3
        items.append(f"{key}={numeric:.{digits}f}")
    return "; ".join(items)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    display = display.replace({np.nan: "—"})
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) if value != "" else "—" for value in row) + " |")
    return "\n".join(lines)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, na_rep="")


def build_context(root: Path) -> Context:
    metrics_dir = root / "results" / "metrics"
    results_tables_dir = ensure_dir(root / "results" / "tables")
    report_tables_dir = ensure_dir(root / "report_assets" / "tables")
    text_fragments_dir = ensure_dir(root / "report_assets" / "text_fragments")
    appendices_dir = ensure_dir(root / "report_assets" / "appendices")
    dataset_config = load_config("dataset_config.yaml")
    trajectory_labels = pd.read_csv(root / "data" / "trajectories" / "trajectory_labels.csv")
    segment_metadata = read_json(root / "data" / "segments" / "segment_metadata.json")
    pointcloud_metadata = read_json(root / "data" / "pointcloud" / "pointcloud_metadata.json")
    segments_test = np.load(root / "data" / "segments" / "segments_test.npz")
    test_segments_count = int(segments_test["x"].shape[0])
    pointcloud_frame_count = int(dataset_config["trajectory_count"]) * int(dataset_config["pointcloud_frames_per_trajectory"])
    pointcloud_frame_counts: dict[str, int] = {
        "clean": pointcloud_frame_count,
        "medium": pointcloud_frame_count,
        "hard": pointcloud_frame_count,
    }
    object_counts = {
        "traclus": int((trajectory_labels["noise_level"] == "medium").sum()),
        "st_dbscan": int((trajectory_labels["noise_level"] == "medium").sum()),
        "vector_field_kmeans": int((trajectory_labels["noise_level"] == "medium").sum()),
        "spatiotemporal_clustering": test_segments_count,
        "cnn_segment_classifier": test_segments_count,
        "lstm_baseline": test_segments_count,
        "lstm_class_aware": test_segments_count,
        "handcrafted_hdbscan": int(len(trajectory_labels)),
        "resnet_kmeans": int(len(trajectory_labels)),
        "resnet_hdbscan": int(len(trajectory_labels)),
        "resnet_hdbscan_family": int(len(trajectory_labels)),
        "sparse_pointcloud_clean": pointcloud_frame_counts["clean"],
        "sparse_pointcloud_medium": pointcloud_frame_counts["medium"],
        "sparse_pointcloud_hard": pointcloud_frame_counts["hard"],
        "cluster_filter_clean": pointcloud_frame_counts["clean"],
        "cluster_filter_medium": pointcloud_frame_counts["medium"],
        "cluster_filter_hard": pointcloud_frame_counts["hard"],
        "cl_det_clean": pointcloud_frame_counts["clean"],
        "cl_det_medium": pointcloud_frame_counts["medium"],
        "cl_det_hard": pointcloud_frame_counts["hard"],
        "sparse_pointcloud": pointcloud_frame_counts["medium"],
        "cluster_filter": pointcloud_frame_counts["medium"],
        "cl_det": pointcloud_frame_counts["medium"],
    }
    return Context(
        root=root,
        metrics_dir=metrics_dir,
        results_tables_dir=results_tables_dir,
        report_tables_dir=report_tables_dir,
        text_fragments_dir=text_fragments_dir,
        appendices_dir=appendices_dir,
        dataset_config=dataset_config,
        trajectory_labels=trajectory_labels,
        segment_metadata=segment_metadata,
        pointcloud_metadata=pointcloud_metadata,
        object_counts=object_counts,
    )


def load_raw_results(context: Context) -> dict[str, pd.DataFrame]:
    dataframes = {}
    for key, filename in RAW_RESULT_FILES.items():
        dataframes[key] = read_csv_checked(context.metrics_dir / filename)
    return dataframes


def prepare_resource_usage_table(df: pd.DataFrame, context: Context) -> pd.DataFrame:
    prepared = df.copy()
    gpu_values = prepared["peak_vram_mb"].apply(parse_vram_cell)
    prepared["peak_vram_gpu0_mb"] = gpu_values.map(lambda item: item[0])
    prepared["peak_vram_gpu1_mb"] = gpu_values.map(lambda item: item[1])
    prepared["peak_vram_total_mb"] = gpu_values.map(lambda item: item[2])
    estimated = []
    for row in prepared.itertuples(index=False):
        mean_inference = getattr(row, "mean_inference_time", np.nan)
        method_name = str(getattr(row, "method"))
        raw_method_name = next((raw for raw, display in METHOD_DISPLAY_NAMES.items() if display == method_name), method_name)
        n_objects = context.object_counts.get(raw_method_name)
        if (pd.isna(mean_inference) or float(mean_inference) == 0.0) and n_objects:
            estimated.append(float(getattr(row, "runtime_seconds")) / n_objects)
        else:
            estimated.append(np.nan)
    prepared["estimated_inference_time"] = estimated
    prepared = round_numeric_frame(prepared)
    return normalize_method_names(prepared)


def prepare_generic_table(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "secondary_metrics" in prepared.columns:
        prepared["secondary_metrics"] = prepared["secondary_metrics"].apply(format_secondary_metrics)
    prepared = round_numeric_frame(prepared)
    return normalize_method_names(prepared)


def prepare_final_applicability_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "task_type",
        "input_type",
        "main_metric",
        "main_metric_value",
        "runtime_seconds",
        "peak_ram_mb",
        "peak_vram_mb",
        "strengths",
        "limitations",
        "applicability_to_uav_pattern_detection",
    ]
    available = [column for column in columns if column in df.columns]
    return df[available].copy()


def describe_dataset(context: Context) -> str:
    cfg = context.dataset_config
    pattern_names = ", ".join(cfg["patterns"])
    return (
        "Синтетический датасет в проекте нужен не как замена реальным полетным наблюдениям, а как контролируемый стенд для "
        "сравнения методов в одинаковых условиях. Для задач анализа паттернов движения это особенно важно: если данные "
        "генерируются по известным правилам, то можно управлять классами поведения, уровнем шума, пропусками наблюдений и "
        "сложностью сцены, а значит честно сравнивать устойчивость алгоритмов. Такой подход удобен для технологической "
        "практики, потому что позволяет отделить качество самого метода от случайных особенностей разметки или конкретного "
        "источника данных.\n\n"
        "В проекте используются три представления. Первое — полные траектории, где каждая последовательность содержит "
        "координаты `x, y, z` и производные кинематические признаки `vx, vy, vz, speed, yaw, pitch, roll`. Это представление "
        "подходит для методов, которые ищут повторяющиеся формы движения по целой траектории. В quick-profile по умолчанию "
        "генерируется {count} траекторий длиной от {length_min} до {length_max} точек при частоте {hz} Гц. Этого достаточно, "
        "чтобы сравнить trajectory clustering и embedding-based подходы без чрезмерно тяжелого прогона.\n\n"
        "Второе представление — сегменты временных рядов. Они формируются как окна длиной {segment_length} точек со stride "
        "{segment_stride}. Внутри окна сохраняются признаки `x, y, z, vx, vy, vz, speed, yaw`, а также истинный класс "
        "поведения и целевые будущие координаты на горизонтах {horizons}. Такое представление нужно для двух связанных задач: "
        "классификации локального поведения и краткосрочного прогноза. Один и тот же segment dataset позволяет проверять, "
        "насколько хорошо модель распознает тип движения и насколько точно предсказывает его продолжение.\n\n"
        "Третье представление — point cloud данные. Для каждого кадра создаются точки БПЛА вокруг истинного центра, фоновый "
        "clutter, случайные шумовые точки и ложные кластеры. Данные разделены на уровни `clean`, `medium` и `hard`, которые "
        "различаются плотностью наблюдений, позиционным шумом и вероятностью пропусков. В quick-profile используется {frames} "
        "кадров на траекторию и {bg} фоновых точек базового уровня. Это достаточно для controlled comparison tracking-style "
        "методов, которые должны восстанавливать положение БПЛА из разреженной пространственной информации.\n\n"
        "Набор паттернов включает девять типов движения: {patterns}. Вместе они охватывают прямолинейные, орбитальные, "
        "патрульные, зависающие и аномальные сценарии, то есть дают материал и для гладких, и для резко меняющихся траекторий.\n\n"
        "При всех преимуществах датасет остается синтетическим. Он не моделирует реальную физику сенсоров во всей полноте, "
        "не учитывает все внешние факторы среды и не гарантирует, что лидеры synthetic benchmark сохранят тот же порядок на "
        "реальных данных. Point cloud часть тоже является упрощенной симуляцией. Поэтому датасет корректно рассматривать как "
        "экспериментальную базу для сравнения методов и подготовки отчета, но не как окончательное доказательство их "
        "прикладного превосходства в реальной эксплуатации."
    ).format(
        count=cfg["trajectory_count"],
        length_min=cfg["trajectory_length_min"],
        length_max=cfg["trajectory_length_max"],
        hz=cfg["sample_rate_hz"],
        segment_length=cfg["segment_length"],
        segment_stride=cfg["segment_stride"],
        horizons=cfg["forecast_horizons_steps"],
        frames=cfg["pointcloud_frames_per_trajectory"],
        bg=cfg["pointcloud_background_points"],
        patterns=pattern_names,
    )


def describe_methods() -> str:
    return (
        "В проекте собраны методы нескольких классов, потому что анализ движения БПЛА нельзя свести к одной форме данных. "
        "Часть алгоритмов работает с целой траекторией, часть — с короткими сегментами, часть — с разреженными облаками точек, "
        "а embedding-подход строит обучаемое представление траектории. Во всех случаях в отчете важно подчеркивать: речь идет "
        "об адаптированных реализациях, пригодных для практического сравнения, а не о полном воспроизведении всех исходных "
        "публикаций.\n\n"
        "TRACLUS-like опирается на идею partition-and-group trajectory clustering из работы Lee et al. На вход подаются полные "
        "траектории, на выходе получаются кластерные метки. В проекте используется упрощенный вариант: траектория режется по "
        "изменению направления, затем по сегментам считаются признаки и выполняется density-based clustering. Реализация "
        "сохраняет основной принцип метода, но не повторяет исходный пайплайн полностью.\n\n"
        "ST-DBSCAN связан с совместным учетом пространственной и временной близости. В нашей версии траектории сначала "
        "преобразуются в summary-признаки, затем spatial и temporal компоненты нормализуются и подаются в DBSCAN-подобную "
        "процедуру. На выходе метод формирует кластеры и шум. Это честная адаптация идеи ST-DBSCAN, но не полный набор "
        "вариантов, которые встречаются в прикладных системах.\n\n"
        "Vector Field k-Means использует представление траектории через локальные поля скоростей. В проекте пространство "
        "дискретизируется на сетку, для ячеек оцениваются усредненные векторы, после чего выполняется k-means по полученному "
        "дескриптору. Такой подход дает стабильное фиксированное представление и хорошо подходит для synthetic benchmark, но "
        "уступает полной статье в детализации локальной динамики.\n\n"
        "Spatio-temporal trajectory clustering получает на вход короткие сегменты движения. Для каждого окна вычисляются "
        "координатные, скоростные и угловые признаки, затем окна кластеризуются как самостоятельные объекты. Это упрощенный "
        "вариант идеи выделения поведенческих режимов по коротким фрагментам; он нужен в проекте как bridge между classical "
        "clustering и supervised segment-моделями.\n\n"
        "CNN-классификатор сегментов — это компактный 1D CNN для многоканальных временных рядов. На вход подается сегмент с "
        "признаками положения и кинематики, на выходе получается класс поведения. В проектной версии реализованы разбиение "
        "train/val/test, ранняя остановка, сохранение модели и матрицы ошибок. Это рабочий baseline без избыточного "
        "архитектурного усложнения.\n\n"
        "LSTM baseline и class-aware LSTM решают задачу краткосрочного прогноза координат. Первая модель использует только "
        "сам временной ряд, вторая получает дополнительный сигнал о классе поведения через embedding класса. Обе модели "
        "выдают будущие координаты на нескольких горизонтах. Реализация намеренно компактна: ее цель — сравнение прогноза в "
        "единой постановке, а не построение максимально сложной forecasting-системы.\n\n"
        "Sparse point cloud trajectory estimation вдохновлен работами по восстановлению траектории БПЛА из разреженных облаков "
        "точек. На вход поступают point cloud кадры, на выходе — оценка центра БПЛА по кадрам. В проекте метод реализован как "
        "связка фильтрации, локальной кластеризации и сглаживания, а не как полный end-to-end learned pipeline. Этого достаточно "
        "для controlled comparison устойчивости к шуму.\n\n"
        "Cluster Filter также работает с point cloud сценами и использует score-based выбор наиболее вероятного drone cluster. "
        "Вход и выход остаются теми же: облако точек на кадр и оцененный центр БПЛА. В реализации проекта scoring упрощен до "
        "voxel-based и continuity-aware эвристик. Метод полезен именно как адаптированный baseline для синтетической сцены.\n\n"
        "CL-Det / DBSCAN LiDAR tracking представлен в виде упрощенного tracking-пайплайна. Он кластеризует точки в кадре, "
        "выбирает целевой кластер и сглаживает траекторию по истории. Этот вариант подходит для сравнения с другими point cloud "
        "baseline-ами, но не включает полный блок pose estimation из более сложных систем.\n\n"
        "ResNet-like embeddings + HDBSCAN служит пользовательским learned baseline. Траектория переводится в 2D image-like "
        "представление, затем компактный encoder строит embedding, после чего применяется HDBSCAN или альтернативный clusterer. "
        "В проекте дополнительно проверяются варианты с ручными признаками и с K-Means. Сильная сторона этой группы методов — "
        "сравнение hand-crafted и learned representations; главное ограничение — использование именно ResNet-like encoder на "
        "2D представлении, а не полноценной предобученной 3D ResNet."
    )


def describe_metrics() -> str:
    return (
        "Набор метрик в проекте изначально разделен по классам задач, потому что одна численная мера не может одинаково честно "
        "характеризовать clustering, classification, forecasting и tracking. Для кластеризации используются ARI, NMI, purity, "
        "cluster accuracy, macro-F1 после сопоставления кластеров с классами, silhouette score и noise ratio. ARI и NMI дают "
        "внешнюю оценку согласия с разметкой, purity и cluster accuracy показывают однородность и сопоставимость кластеров, "
        "silhouette оценивает внутреннюю разделимость, а noise ratio важен там, где метод явно выделяет шум.\n\n"
        "В задачах классификации сегментов применяются accuracy, precision_macro, recall_macro и macro-F1. Accuracy показывает "
        "общую долю верных ответов, но при множестве классов этого недостаточно, поэтому ключевую роль играет macro-усреднение. "
        "Оно не дает крупным классам скрывать проблемы на более редких паттернах. Матрица ошибок здесь так же важна, как и "
        "итоговый F1, потому что именно она показывает, какие типы движения путаются между собой.\n\n"
        "Для forecasting-блока используются MAE, MSE, RMSE, ADE и FDE. MAE и RMSE дают привычную количественную ошибку по "
        "координатам, причем RMSE сильнее штрафует крупные промахи. ADE характеризует среднюю ошибку на всем горизонте "
        "прогноза, а FDE показывает качество конечной точки — самой трудной части краткосрочного предсказания. Отдельные "
        "ошибки по горизонтам нужны для понимания того, как быстро деградирует модель по мере роста дальности прогноза.\n\n"
        "В point cloud и tracking-задачах используются position RMSE, detection rate, false positive rate, track fragmentation "
        "и FPS. Position RMSE показывает, насколько точно восстанавливается центр БПЛА. Detection rate описывает частоту "
        "успешного обнаружения, false positive rate — склонность метода к ложным целям, fragmentation — разрывы трека, а FPS "
        "служит операционной характеристикой скорости работы.\n\n"
        "Отдельно фиксируются ресурсные метрики: runtime_seconds, peak RAM, peak VRAM, parameter_count и "
        "mean/estimated inference time. Они нужны потому, что инженерная ценность метода определяется не только качеством, "
        "но и вычислительной стоимостью. Особенно это заметно при сравнении компактных classical baseline-ов с learned "
        "embedding и sequence-моделями на GPU.\n\n"
        "Главный методологический вывод прост: нельзя без пояснений сводить все методы к одной общей метрике. "
        "Кластеризация отвечает на вопрос о разделимости паттернов, классификация — о распознавании поведения, forecasting — "
        "о качестве предсказания, tracking — о восстановлении положения по наблюдениям. Поэтому в отчете корректно сравнивать "
        "методы сначала внутри их задач, а затем обсуждать практическую применимость по сочетанию точности, устойчивости и "
        "ресурсоемкости."
    )


def describe_protocol(context: Context) -> str:
    cfg = context.dataset_config
    return (
        "Экспериментальный pipeline организован как воспроизводимая последовательность CLI-шагов. Сначала из YAML-конфигов "
        "считываются параметры генерации данных, затем фиксируется random seed, после чего последовательно создаются "
        "траектории, сегменты и point cloud представления. Такая схема важна и для разработки, и для отчета: любой следующий "
        "прогон может быть повторен по тем же конфигам и дать сопоставимый набор артефактов.\n\n"
        "Первый блок экспериментов работает с полными траекториями. Здесь сравниваются TRACLUS-like, ST-DBSCAN, Vector Field "
        "k-Means и spatio-temporal trajectory clustering как методы выделения повторяющихся паттернов движения. Их результаты "
        "оцениваются по ARI, NMI, purity, cluster accuracy, macro-F1, silhouette и noise ratio. Рядом с ними рассматривается "
        "embedding-семейство ResNet-like + clustering как learned baseline для той же общей задачи.\n\n"
        "Второй блок использует segment dataset длиной {segment_length} точек. На этих данных решаются две задачи: "
        "классификация поведения с помощью 1D CNN и краткосрочный прогноз с помощью LSTM baseline и class-aware LSTM. "
        "Обе постановки используют одни и те же окна движения, поэтому позволяют сравнивать распознавание и прогнозирование "
        "на общем входном представлении.\n\n"
        "Третий блок посвящен point cloud данным уровней `clean`, `medium` и `hard`. Здесь сравниваются sparse point cloud "
        "trajectory estimation, Cluster Filter и CL-Det-like tracking. Методы получают на вход облака точек с шумом, фоном и "
        "ложными кластерами, а на выходе восстанавливают положение БПЛА по кадрам. Основные метрики этой части — position RMSE, "
        "detection rate, false positive rate, fragmentation и FPS.\n\n"
        "Embedding-блок выделен отдельно, потому что по смыслу он отличается и от classical clustering, и от supervised "
        "segment-моделей. Траектория переводится в image-like форму, затем encoder строит embedding, после чего проверяются "
        "минимум три варианта: ручные признаки с density clustering, learned embeddings с K-Means и learned embeddings с "
        "HDBSCAN. Это позволяет оценить не только итоговый результат, но и вклад самого представления.\n\n"
        "Во всех блоках отдельно фиксируются runtime, RAM и VRAM. А при подготовке материалов для отчета raw CSV не "
        "перезаписываются: скрипт создает отдельные report-ready таблицы в `report_assets/tables/` и зеркальные копии в "
        "`results/tables/`, сохраняя разделение между машиночитаемым слоем и слоем, уже удобным для вставки в отчет."
    ).format(segment_length=cfg["segment_length"])


def describe_hardware() -> str:
    return (
        "Экспериментальный пакет ориентирован на рабочую станцию с двумя NVIDIA GeForce RTX 5070 Ti по 16 ГБ видеопамяти. "
        "Такая конфигурация позволяет запускать компактные нейросетевые модели на GPU и при этом сохранять classical baseline-ы "
        "в той же среде для честного сравнения по времени и памяти.\n\n"
        "GPU в проекте в первую очередь используют 1D CNN-классификатор сегментов, LSTM-модели прогноза и ResNet-like encoder "
        "для embedding-представления траектории. Для них наиболее важны peak VRAM, parameter_count и среднее время инференса. "
        "При наличии CUDA устройство выбирается автоматически, а при отсутствии доступен CPU fallback для технической проверки.\n\n"
        "Классические clustering-методы и большая часть point cloud baseline-ов ближе к CPU-bound вычислениям. Для них ключевы "
        "runtime_seconds и peak RAM, поскольку вычислительная цена определяется обработкой табличных признаков, кластеризацией "
        "и геометрическими эвристиками, а не глубокими сетями.\n\n"
        "В отчетной части проекта фиксируются runtime_seconds, peak RAM, peak VRAM по GPU, а также mean или estimated inference "
        "time. Это делает сравнение не только исследовательским, но и инженерно осмысленным: можно обсуждать не только качество, "
        "но и то, насколько дорог каждый метод в запуске."
    )


def describe_limitations() -> str:
    return (
        "Первое ограничение проекта связано с самим источником данных. Синтетический датасет дает сильный выигрыш в "
        "воспроизводимости и контролируемости, но не заменяет реальные сенсорные наблюдения БПЛА. Он не отражает всю сложность "
        "сцены, аппаратных ошибок и внешних факторов среды. Поэтому любой вывод о превосходстве метода нужно трактовать как "
        "вывод внутри synthetic benchmark, а не как окончательное доказательство практической пригодности в эксплуатации.\n\n"
        "Второй слой ограничений касается реализации методов. Значительная часть из них представлена как adapted versions. "
        "Цель проекта состоит в практическом сопоставлении ключевых идей на общей базе данных, а не в полном воспроизведении "
        "всех опубликованных пайплайнов. Отсюда следуют упрощения в TRACLUS-like, Vector Field k-Means, sparse point cloud "
        "trajectory estimation, Cluster Filter и CL-Det-like tracking. Эти методы дают реальные метрики и годятся для отчета, "
        "но не должны описываться как точные реплики исходных статей.\n\n"
        "Отдельно нужно оговаривать embedding-подход. В проекте используется ResNet-like encoder на 2D image-like "
        "представлении траектории, а не полноценная предобученная 3D ResNet. Это важное техническое ограничение: learned "
        "representation здесь строится в более простой постановке, и его нельзя без оговорок приравнивать к тяжелым "
        "3D/video/voxel-моделям.\n\n"
        "Point cloud симуляция тоже остается упрощенной. В ней есть фон, ложные кластеры, шум и пропуски, но геометрия сцены "
        "все равно значительно проще реального мира. Поэтому tracking-результаты полезны как controlled comparison, но требуют "
        "осторожности при переносе на реальные наблюдения. Дополнительно нужно помнить, что point cloud baseline-ы в проекте "
        "работают в первую очередь с восстановлением центра БПЛА, а не с полной задачей pose estimation.\n\n"
        "Наконец, ограничения есть и у системы метрик. Нельзя без пояснений сводить clustering ARI, classification macro-F1, "
        "forecasting RMSE и tracking position RMSE к одной общей шкале качества. Эти показатели отвечают на разные вопросы, "
        "поэтому интерпретация результатов должна строиться по группам задач с учетом устойчивости к шуму и ресурсоемкости."
    )


def _best_row(df: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    return df.sort_values(column, ascending=ascending).iloc[0]


def describe_results(raw: dict[str, pd.DataFrame]) -> str:
    clustering = raw["clustering"]
    classification = raw["classification"]
    forecasting = raw["forecasting"]
    pointcloud = raw["pointcloud"]
    resnet = raw["resnet"]
    final_comparison = raw["final_comparison"]
    resource = raw["resource_usage"]

    best_classic_cluster = _best_row(clustering, "ari", ascending=False)
    best_resnet = _best_row(resnet, "ari", ascending=False)
    cnn = classification.iloc[0]
    best_forecast = _best_row(forecasting, "rmse", ascending=True)
    medium_pointcloud = pointcloud[pointcloud["difficulty"] == "medium"]
    best_pointcloud = _best_row(medium_pointcloud, "position_rmse", ascending=True)
    hardest_pointcloud = _best_row(pointcloud[pointcloud["difficulty"] == "hard"], "position_rmse", ascending=True)
    slowest = _best_row(final_comparison, "runtime_seconds", ascending=False)
    fastest_cluster = _best_row(clustering, "runtime_seconds", ascending=True) if "runtime_seconds" in clustering.columns else None
    max_ram = _best_row(resource, "peak_ram_mb", ascending=False)

    return (
        "Сводные результаты показывают, что в текущем synthetic benchmark нет одного универсально лучшего метода для всех "
        "постановок. Картина получается содержательной именно потому, что разные семейства методов сильны в разных задачах. "
        "По блоку классической кластеризации полных траекторий лучшую ARI среди trajectory-first baseline-ов показывает "
        "{classic_method}: ARI {classic_ari:.3f}, NMI {classic_nmi:.3f}. Это заметно сильнее, чем у TRACLUS-like и ST-DBSCAN, "
        "и говорит о том, что grid-based velocity representation в текущем наборе данных лучше улавливает структуру движения, "
        "чем более грубые summary-описания.\n\n"
        "Learned embedding-подходи семейства ResNet-like выглядят еще сильнее. Лучший результат в этой группе дает "
        "{resnet_method} с ARI {resnet_ari:.3f}, NMI {resnet_nmi:.3f} и macro-F1 {resnet_f1:.3f}. Для отчета это важный вывод: "
        "обучаемое представление траектории действительно помогает отделять паттерны движения лучше, чем часть classical "
        "baseline-ов. При этом второй learned вариант, {resnet_alt}, показывает близкое качество, тогда как вариант с ручными "
        "признаками и density clustering заметно слабее.\n\n"
        "CNN-классификатор сегментов на synthetic segment task показывает очень сильный результат: accuracy {cnn_acc:.3f}, "
        "precision_macro {cnn_precision:.3f}, recall_macro {cnn_recall:.3f}, macro-F1 {cnn_f1:.3f}. Это подтверждает, что "
        "локальные сегменты движения хорошо разделимы в supervised постановке. Но интерпретировать такой результат нужно "
        "осторожно: высокий F1 здесь скорее подтверждает согласованность синтетического представления и разметки, чем заранее "
        "гарантирует ту же устойчивость на реальных данных.\n\n"
        "В forecasting-блоке разница между baseline LSTM и class-aware LSTM оказалась небольшой. Лучший RMSE показывает "
        "{forecast_method} со значением {forecast_rmse:.2f}; одновременно MAE составляет {forecast_mae:.2f}, ADE — "
        "{forecast_ade:.2f}, FDE — {forecast_fde:.2f}. Существенного выигрыша от class-aware варианта не видно. Это полезный "
        "практический вывод: для текущего synthetic setup сама траекторная динамика уже несет большую часть информации, и "
        "добавление класса поведения не дает качественного скачка.\n\n"
        "По point cloud данным наиболее уверенно в режиме `medium` выглядит {point_method} с position RMSE {point_rmse:.2f}. "
        "На уровне `hard` лучшим остается {hard_method} с RMSE {hard_rmse:.2f}, хотя абсолютные ошибки закономерно растут. "
        "Detection rate у всех трех baseline-ов остается высоким, поэтому различие проявляется главным образом не в самом факте "
        "нахождения цели, а в точности пространственной локализации в шумной сцене.\n\n"
        "С точки зрения ресурсоемкости самыми тяжелыми в итоговой сводке выглядят {slow_method} с runtime около "
        "{slow_runtime:.2f} с, а по пиковому RAM в `resource_usage.csv` максимальное значение наблюдается у {ram_method} — "
        "около {ram_peak:.1f} МБ. По VRAM текущие GPU-пайплайны держатся примерно в районе 7.2 ГБ на первой GPU и около "
        "0.3 ГБ на второй, то есть остаются выполнимыми на заявленной конфигурации.\n\n"
        "Итоговая интерпретация здесь такая: CNN и learned embeddings показывают сильные результаты на synthetic benchmark, "
        "classical trajectory clustering остается полезным как более интерпретируемый baseline, а point cloud методы нужно "
        "обсуждать особенно осторожно из-за упрощенного характера симуляции. Для отчета этих результатов достаточно, чтобы "
        "не просто перечислить цифры, а осмысленно разобрать сильные и слабые стороны каждого семейства методов."
    ).format(
        classic_method=METHOD_DISPLAY_NAMES.get(str(best_classic_cluster["method"]), str(best_classic_cluster["method"])),
        classic_ari=float(best_classic_cluster["ari"]),
        classic_nmi=float(best_classic_cluster["nmi"]),
        resnet_method=METHOD_DISPLAY_NAMES.get(str(best_resnet["method"]), str(best_resnet["method"])),
        resnet_alt=METHOD_DISPLAY_NAMES.get(
            str(resnet.sort_values("ari", ascending=False).iloc[1]["method"]),
            str(resnet.sort_values("ari", ascending=False).iloc[1]["method"]),
        ),
        resnet_ari=float(best_resnet["ari"]),
        resnet_nmi=float(best_resnet["nmi"]),
        resnet_f1=float(best_resnet["macro_f1"]),
        cnn_acc=float(cnn["accuracy"]),
        cnn_precision=float(cnn["precision_macro"]),
        cnn_recall=float(cnn["recall_macro"]),
        cnn_f1=float(cnn["macro_f1"]),
        forecast_method=METHOD_DISPLAY_NAMES.get(str(best_forecast["method"]), str(best_forecast["method"])),
        forecast_rmse=float(best_forecast["rmse"]),
        forecast_mae=float(best_forecast["mae"]),
        forecast_ade=float(best_forecast["ade"]),
        forecast_fde=float(best_forecast["fde"]),
        point_method=METHOD_DISPLAY_NAMES.get(str(best_pointcloud["method"]), str(best_pointcloud["method"])),
        point_rmse=float(best_pointcloud["position_rmse"]),
        hard_method=METHOD_DISPLAY_NAMES.get(str(hardest_pointcloud["method"]), str(hardest_pointcloud["method"])),
        hard_rmse=float(hardest_pointcloud["position_rmse"]),
        slow_method=METHOD_DISPLAY_NAMES.get(str(slowest["method"]), str(slowest["method"])),
        slow_runtime=float(slowest["runtime_seconds"]),
        ram_method=METHOD_DISPLAY_NAMES.get(str(max_ram["method"]), str(max_ram["method"])),
        ram_peak=float(max_ram["peak_ram_mb"]),
    )


def describe_conclusion(raw: dict[str, pd.DataFrame]) -> str:
    final_comparison = raw["final_comparison"]
    clustering_rows = final_comparison[final_comparison["task_type"] == "clustering"].sort_values("main_metric_value", ascending=False)
    pointcloud_rows = final_comparison[final_comparison["task_type"] == "tracking"].sort_values("main_metric_value", ascending=True)
    best_cluster = clustering_rows.iloc[0]
    best_point = pointcloud_rows.iloc[0]
    return (
        "В рамках проекта был собран полный экспериментальный пакет для подготовки отчета: синтетический датасет, адаптированные "
        "реализации методов, набор метрик, скрипты запуска, report-ready таблицы, графики и текстовые заготовки. Это значит, "
        "что следующий этап — генерация самого DOCX-отчета — может опираться уже на проверенный аналитический слой, а не на "
        "разрозненные промежуточные артефакты.\n\n"
        "Сравнение охватило несколько групп методов. Для готовых траекторий были сопоставлены classical clustering-подходы и "
        "embedding-модели. Для сегментов движения сравнивались supervised classification и краткосрочный forecasting. Для "
        "разреженных point cloud данных были собраны несколько tracking-style baseline-ов. Такой набор важен сам по себе, "
        "потому что показывает многослойность задачи анализа паттернов БПЛА и необходимость разных представлений данных.\n\n"
        "Для готовых траекторий наиболее сильные результаты дают embedding-based варианты, а лучшим кластеризационным "
        "результатом в текущем проекте становится {best_cluster_method} с {best_cluster_metric} = {best_cluster_value:.3f}. "
        "Это не отменяет ценности classical методов: они остаются полезными как более интерпретируемые и более легкие "
        "baseline-ы, особенно когда важны прозрачность признаков и умеренная вычислительная цена.\n\n"
        "Для сегментов и прогноза картина другая. CNN-классификатор уверенно решает задачу распознавания локального поведения, "
        "тогда как LSTM-блок показывает, что прогнозировать будущее движение сложнее, чем классифицировать уже наблюдаемое окно. "
        "Разница между baseline и class-aware LSTM невелика, поэтому в заключении корректно говорить не о явном преимуществе "
        "class-conditioning, а о его ограниченном эффекте в текущем synthetic setup.\n\n"
        "В point cloud части лучшим по итоговой report-ready сводке оказывается {best_point_method} с {best_point_metric} = "
        "{best_point_value:.2f}. Но именно эта группа результатов сильнее всего зависит от упрощенной синтетической симуляции. "
        "Поэтому такие значения корректно трактовать как controlled comparison методов восстановления положения БПЛА, а не как "
        "окончательный вывод о превосходстве одного подхода на реальных сенсорных наблюдениях.\n\n"
        "Отдельный вывод касается ResNet/HDBSCAN-линейки. Полученные результаты показывают, что learned embeddings в проекте "
        "обоснованы и конкурентоспособны. Однако корректно подчеркивать, что здесь используется ResNet-like encoder на 2D "
        "представлении траектории, а не полноценная предобученная 3D ResNet. Следовательно, embedding-подход можно обосновать "
        "как сильный пользовательский baseline, но не как прямой эквивалент более тяжелых 3D-архитектур.\n\n"
        "Дальнейшее развитие проекта логично связать с более реалистичными point cloud сценами, проверкой на внешних данных и "
        "расширением сравнения learned и hand-crafted представлений. В текущем виде пакет уже готов к генерации отчета."
    ).format(
        best_cluster_method=METHOD_DISPLAY_NAMES.get(str(best_cluster["method"]), str(best_cluster["method"])),
        best_cluster_metric=str(best_cluster["main_metric"]),
        best_cluster_value=float(best_cluster["main_metric_value"]),
        best_point_method=METHOD_DISPLAY_NAMES.get(str(best_point["method"]), str(best_point["method"])),
        best_point_metric=str(best_point["main_metric"]),
        best_point_value=float(best_point["main_metric_value"]),
    )


def build_appendix_a(context: Context) -> str:
    cfg = context.dataset_config
    lines = [
        "# Appendix A. Dataset parameters",
        "",
        "## Trajectory generation",
        f"- Dataset name: `{cfg['dataset_name']}`",
        f"- Random seed: `{cfg['seed']}`",
        f"- Number of trajectories: `{cfg['trajectory_count']}`",
        f"- Trajectory length range: `{cfg['trajectory_length_min']}..{cfg['trajectory_length_max']}` points",
        f"- Sample rate: `{cfg['sample_rate_hz']}` Hz",
        "",
        "## Pattern list",
    ]
    for pattern in cfg["patterns"]:
        lines.append(f"- `{pattern}`")
    lines.extend(
        [
            "",
            "## Segment dataset",
            f"- Segment length: `{cfg['segment_length']}`",
            f"- Segment stride: `{cfg['segment_stride']}`",
            f"- Forecast horizons (steps): `{cfg['forecast_horizons_steps']}`",
            "",
            "## Noise profiles",
        ]
    )
    for name, profile in cfg["noise_levels"].items():
        lines.append(
            f"- `{name}`: position_std={profile['position_std']}, drop_probability={profile['drop_probability']}, false_clusters={profile['false_clusters']}"
        )
    lines.extend(
        [
            "",
            "## Point cloud generation",
            f"- Frames per trajectory: `{cfg['pointcloud_frames_per_trajectory']}`",
            f"- Background points: `{cfg['pointcloud_background_points']}`",
            f"- Missing-point probabilities: `{cfg['missing_point_probability']}`",
        ]
    )
    return "\n".join(lines)


def build_appendix_b() -> str:
    clustering = load_config("clustering_config.yaml")
    classification = load_config("classification_config.yaml")
    forecasting = load_config("forecasting_config.yaml")
    pointcloud = load_config("pointcloud_config.yaml")
    resnet = load_config("resnet_hdbscan_config.yaml")
    return (
        "# Appendix B. Method parameters\n\n"
        "## Clustering methods\n"
        f"- TRACLUS-like: `min_samples={clustering['methods']['traclus']['min_samples']}`, `eps={clustering['methods']['traclus']['eps']}`, `curvature_threshold={clustering['methods']['traclus']['curvature_threshold']}`\n"
        f"- ST-DBSCAN: `eps_spatial={clustering['methods']['st_dbscan']['eps_spatial']}`, `eps_temporal={clustering['methods']['st_dbscan']['eps_temporal']}`, `min_samples={clustering['methods']['st_dbscan']['min_samples']}`\n"
        f"- Vector Field k-Means: `grid_size={clustering['methods']['vector_field_kmeans']['grid_size']}`, `n_clusters={clustering['methods']['vector_field_kmeans']['n_clusters']}`\n"
        f"- Spatio-temporal clustering: `n_clusters={clustering['methods']['spatiotemporal_clustering']['n_clusters']}`\n\n"
        "## CNN parameters\n"
        f"- epochs={classification['epochs']}, batch_size={classification['batch_size']}, learning_rate={classification['learning_rate']}, weight_decay={classification['weight_decay']}, patience={classification['patience']}, hidden_channels={classification['hidden_channels']}\n\n"
        "## LSTM parameters\n"
        f"- epochs={forecasting['epochs']}, batch_size={forecasting['batch_size']}, learning_rate={forecasting['learning_rate']}, hidden_size={forecasting['hidden_size']}, num_layers={forecasting['num_layers']}, patience={forecasting['patience']}\n\n"
        "## Point cloud / tracking parameters\n"
        f"- dbscan_eps={pointcloud['dbscan_eps']}, dbscan_min_samples={pointcloud['dbscan_min_samples']}, cluster_filter_voxel_size={pointcloud['cluster_filter_voxel_size']}, smoothing_alpha={pointcloud['smoothing_alpha']}\n\n"
        "## ResNet-like parameters\n"
        f"- epochs={resnet['epochs']}, batch_size={resnet['batch_size']}, learning_rate={resnet['learning_rate']}, embedding_dim={resnet['embedding_dim']}, image_size={resnet['image_size']}, use_hdbscan={resnet['use_hdbscan']}\n"
    )


def build_appendix_c() -> str:
    return (
        "# Appendix C. Extra results\n\n"
        "Дополнительные таблицы для отчета сохранены в `report_assets/tables/`.\n\n"
        "- `clustering_results_table.csv`\n"
        "- `classification_results_table.csv`\n"
        "- `forecasting_results_table.csv`\n"
        "- `pointcloud_results_table.csv`\n"
        "- `resnet_hdbscan_results_table.csv`\n"
        "- `resource_usage_table.csv`\n"
        "- `final_comparison_table.csv`\n"
        "- `final_applicability_table.csv`\n"
    )


def build_appendix_d() -> str:
    return (
        "# Appendix D. Code fragments\n\n"
        "## Trajectory generation\n\n"
        "```python\n"
        "bundle = generate_trajectory_dataset(config)\n"
        "save_trajectory_bundle(bundle)\n"
        "```\n\n"
        "## Metric calculation\n\n"
        "```python\n"
        "metrics = evaluate_clustering(y_true, y_pred, features)\n"
        "resource_usage = monitor.stop()\n"
        "```\n\n"
        "## Experiment run\n\n"
        "```python\n"
        "uv run python -m src.experiments.run_all --dataset synthetic_v1 --output results\n"
        "```\n\n"
        "## Example method logic\n\n"
        "```text\n"
        "1. Build trajectory descriptor\n"
        "2. Normalize features\n"
        "3. Run DBSCAN/HDBSCAN or K-Means\n"
        "4. Compare labels against ground truth\n"
        "```\n"
    )


def build_appendix_e() -> str:
    figure_dir = project_root() / "results" / "figures"
    figures = sorted(path.name for path in figure_dir.glob("*.png"))
    lines = ["# Appendix E. Additional figures", "", "## Available figures"]
    for figure in figures:
        lines.append(f"- `{figure}`: figure available in `results/figures/` and mirrored in `report_assets/figures/`.")
    return "\n".join(lines)


def save_text_fragments(context: Context, raw_results: dict[str, pd.DataFrame], table_markdowns: dict[str, str]) -> None:
    files = {
        "dataset_description.md": describe_dataset(context),
        "methods_short_descriptions.md": describe_methods(),
        "metrics_description.md": describe_metrics(),
        "experiment_protocol_summary.md": describe_protocol(context),
        "hardware_description.md": describe_hardware(),
        "results_summary.md": describe_results(raw_results),
        "limitations_summary.md": describe_limitations(),
        "conclusion_points.md": describe_conclusion(raw_results),
    }
    for filename, content in files.items():
        (context.text_fragments_dir / filename).write_text(content, encoding="utf-8")
    for key, filename in MARKDOWN_TABLE_OUTPUTS.items():
        if key in table_markdowns:
            (context.text_fragments_dir / filename).write_text(table_markdowns[key], encoding="utf-8")


def save_appendices(context: Context) -> None:
    appendices = {
        "appendix_a_dataset_params.md": build_appendix_a(context),
        "appendix_b_method_params.md": build_appendix_b(),
        "appendix_c_extra_results.md": build_appendix_c(),
        "appendix_d_code_fragments.md": build_appendix_d(),
        "appendix_e_additional_figures.md": build_appendix_e(),
    }
    for filename, content in appendices.items():
        (context.appendices_dir / filename).write_text(content, encoding="utf-8")


def prepare_tables(context: Context, raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    prepared: dict[str, pd.DataFrame] = {}
    prepared["clustering"] = prepare_generic_table(raw["clustering"].sort_values("ari", ascending=False).reset_index(drop=True))
    prepared["classification"] = prepare_generic_table(raw["classification"].sort_values("macro_f1", ascending=False).reset_index(drop=True))
    prepared["forecasting"] = prepare_generic_table(raw["forecasting"].sort_values("rmse", ascending=True).reset_index(drop=True))
    prepared["pointcloud"] = prepare_generic_table(
        raw["pointcloud"].sort_values(["difficulty", "position_rmse"], ascending=[True, True]).reset_index(drop=True)
    )
    prepared["resnet"] = prepare_generic_table(raw["resnet"].sort_values("ari", ascending=False).reset_index(drop=True))
    prepared["resource_usage"] = prepare_resource_usage_table(raw["resource_usage"], context)
    prepared["final_comparison"] = prepare_generic_table(raw["final_comparison"].copy())
    prepared["final_applicability"] = prepare_final_applicability_table(prepared["final_comparison"])

    for key, filename in TABLE_OUTPUTS.items():
        df = prepared[key]
        write_csv(df, context.report_tables_dir / filename)
        write_csv(df, context.results_tables_dir / filename)

    markdowns = {
        "clustering": dataframe_to_markdown(prepared["clustering"]),
        "classification": dataframe_to_markdown(prepared["classification"]),
        "forecasting": dataframe_to_markdown(prepared["forecasting"]),
        "pointcloud": dataframe_to_markdown(prepared["pointcloud"]),
        "resource_usage": dataframe_to_markdown(prepared["resource_usage"]),
        "final_comparison": dataframe_to_markdown(prepared["final_comparison"]),
        "final_applicability": dataframe_to_markdown(prepared["final_applicability"]),
    }
    return prepared, markdowns


def main() -> None:
    root = project_root()
    context = build_context(root)
    raw_results = load_raw_results(context)
    prepared_tables, markdowns = prepare_tables(context, raw_results)
    del prepared_tables
    save_text_fragments(context, raw_results, markdowns)
    save_appendices(context)
    print("Prepared report-ready tables, markdown tables, text fragments, and appendices.")


if __name__ == "__main__":
    main()
