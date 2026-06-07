# UAV Pattern Practice

Этот проект собирает экспериментальный пакет для практического сравнения методов анализа паттернов движения БПЛА на синтетических данных. Корневой workflow ориентирован на `uv`: зависимости фиксируются в `pyproject.toml`, запуск выполняется через `uv run ...`.

## Цель проекта

- Сгенерировать синтетический датасет траекторий, сегментов и point cloud данных.
- Реализовать адаптированные методы кластеризации, классификации, прогнозирования и трекинга.
- Получить метрики, графики, таблицы и markdown-материалы для последующей генерации DOCX-отчета.

## Состав методов

- `TRACLUS-like`
- `ST-DBSCAN`
- `Vector Field k-Means`
- `Spatio-temporal clustering`
- `1D CNN segment classifier`
- `LSTM baseline`
- `LSTM class-aware`
- `Sparse pointcloud estimation`
- `Cluster Filter`
- `CL-Det-like tracking`
- `ResNet-like embeddings + HDBSCAN/DBSCAN/KMeans`

## Установка зависимостей

Основной способ:

```bash
uv sync
```

Если нужно добавлять новые библиотеки:

```bash
uv add <package>
```

## Генерация датасета

```bash
uv run python -m src.dataset.generate_trajectories
uv run python -m src.dataset.generate_segments
uv run python -m src.dataset.generate_pointcloud
```

## Запуск всех экспериментов

```bash
uv run python -m src.experiments.run_all --dataset synthetic_v1 --output results
```

## Отдельные запуски

```bash
uv run python -m src.experiments.run_clustering
uv run python -m src.experiments.run_classification
uv run python -m src.experiments.run_forecasting
uv run python -m src.experiments.run_pointcloud
uv run python -m src.experiments.run_resnet_hdbscan
```

## Где лежат результаты

- Метрики: `results/metrics/`
- Логи запусков: `results/logs/`
- Графики: `results/figures/`
- Сводные таблицы: `results/tables/`

## Где лежат материалы для отчета

- Фигуры: `report_assets/figures/`
- Таблицы: `report_assets/tables/`
- Текстовые фрагменты: `report_assets/text_fragments/`
- Приложения: `report_assets/appendices/`
- Шаблон структуры: `report_template/report_structure.md`

## Подготовка материалов для отчета

Команды:

```bash
python -m src.experiments.prepare_report_assets
python -m src.experiments.validate_outputs
```

`prepare_report_assets` читает raw-результаты из `results/metrics/`, не переписывает их и создает отдельные report-ready таблицы в `report_assets/tables/` и зеркальные копии в `results/tables/`. Тот же скрипт пересобирает markdown-таблицы, текстовые фрагменты и приложения для будущего отчета.

`validate_outputs` проверяет документацию, датасет, ключевые CSV, фигуры и report assets, а затем создает `VALIDATION_REPORT.md` в корне проекта. Если в отчете указан статус `READY FOR REPORT GENERATION`, проект можно считать подготовленным к следующему этапу генерации DOCX.

## Адаптированные реализации

- `TRACLUS-like`, `Vector Field k-Means`, `sparse pointcloud estimation`, `Cluster Filter`, `CL-Det-like` и `ResNet-like embeddings` реализованы как адаптированные версии ключевых идей статей.
- Embedding-подход использует 2D представление траекторий и ResNet-like encoder, а не полноценную 3D ResNet.
- Упрощения и ограничения перечислены в `limitations.md`.

## Что уже считается готовым для DOCX

- Сгенерированные датасеты в `data/`
- CSV-метрики и JSON-логи
- PNG-графики для отчета
- Таблицы и текстовые markdown-заготовки
- Сводный `final_comparison.csv`
- `VALIDATION_REPORT.md` со статусом готовности
