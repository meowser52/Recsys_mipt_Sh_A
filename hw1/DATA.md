# Данные

**Студент:** Шарипов Артем

**Датасет:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), категория **Video Games** (McAuley Lab, UCSD).

- страница датасета: <https://amazon-reviews-2023.github.io/>
- отзывы, официальная 5-core выборка `rating_only`: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz>
- метаданные товаров: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz>
- статья: Y. Hou, J. Li, Z. He, A. Yan, X. Chen, J. McAuley. *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952, 2024.

**Объём:** 814 586 отзывов, 94 762 пользователя, 25 612 товаров, период 1999-10 … 2023-09. Поля отзывов: `user_id`, `parent_asin`, `rating` (1–5), `timestamp`. Метаданные: название, путь категорий (платформа, тип товара), `store`, цена, ESRB.

**Как получить:** `experiments.ipynb` сам скачивает оба файла (~116 МБ) в `hw1/data/raw/` и выводит их SHA-256. Вручную:

```bash
mkdir -p data/raw && cd data/raw
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz
```

Данные в репозиторий не загружаются (`data/` в `.gitignore`).
