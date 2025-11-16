import pandas as pd
import numpy as np
import gc
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb
from tqdm.auto import tqdm

# --- 1. Загрузка и подготовка данных ---

print("Загрузка данных...")
# Используем polars для быстрой загрузки, затем конвертируем в pandas
# Если polars не установлен, можно использовать pd.read_parquet('train_data.pq')
try:
    import polars as pl
    train_df = pl.read_parquet('train_data.pq').to_pandas()
except ImportError:
    train_df = pd.read_parquet('train_data.pq')

sample_submission = pd.read_csv('sample_submission.csv')
print("Данные загружены.")
print(f"Размер трейна: {train_df.shape}")


# --- Временное разделение для валидации ---
# Обучаемся на данных до 40-го дня, валидируемся на днях с 40 по 47.
# Это имитирует реальную задачу предсказания последней недели.
MAX_DATE = train_df['date'].max()
VAL_DAYS = 7
train_data = train_df[train_df['date'] <= MAX_DATE - VAL_DAYS]
val_data = train_df[train_df['date'] > MAX_DATE - VAL_DAYS]

print(f"Данные для обучения: {train_data.shape}")
print(f"Данные для валидации: {val_data.shape}")

# Создаем маппинги для user_id и item_id для корректной работы моделей
all_users = train_df['user_id'].unique()
all_items = train_df['item_id'].unique()

user_map = {user_id: i for i, user_id in enumerate(all_users)}
item_map = {item_id: i for i, item_id in enumerate(all_items)}

# Инвертированные маппинги для восстановления исходных ID
inv_user_map = {i: user_id for user_id, i in user_map.items()}
inv_item_map = {i: item_id for item_id, i in item_map.items()}

train_data['user_idx'] = train_data['user_id'].map(user_map)
train_data['item_idx'] = train_data['item_id'].map(item_map)


# --- 2. Этап 1: Генерация кандидатов ---

# Словарь для хранения кандидатов для каждого пользователя
candidates = {}
test_users_idx = [user_map[u] for u in sample_submission['user_id'].unique() if u in user_map]

# --- Генератор 1: ALS ---
print("Обучение ALS модели...")
# Создание sparse-матрицы взаимодействий
user_item_matrix = csr_matrix(
    (np.ones(len(train_data)), (train_data['user_idx'], train_data['item_idx'])),
    shape=(len(all_users), len(all_items))
)

# Обучение модели ALS
# Параметры подобраны для хорошего баланса скорости и качества
als_model = AlternatingLeastSquares(
    factors=128,
    regularization=0.01,
    iterations=15,
    random_state=42,
    use_gpu=False # Установите True, если у вас есть GPU и установлен `implicit-cu`
)
als_model.fit(user_item_matrix)

print("Генерация кандидатов с помощью ALS...")
# `N` - количество кандидатов, которые мы хотим сгенерировать
user_ids_to_recommend = test_users_idx
als_candidates_raw = als_model.recommend(
    user_ids_to_recommend,
    user_item_matrix[user_ids_to_recommend],
    N=100,
    filter_already_liked_items=False
)

for user_idx, item_indices in tqdm(zip(user_ids_to_recommend, als_candidates_raw[0]), total=len(user_ids_to_recommend)):
    user_id = inv_user_map[user_idx]
    candidates.setdefault(user_id, set())
    for item_idx in item_indices:
        candidates[user_id].add(inv_item_map[item_idx])

del als_model, user_item_matrix
gc.collect()

# --- Генератор 2: Популярные товары за последнюю неделю ---
print("Генерация кандидатов на основе недавней популярности...")
LAST_WEEK_DATE = train_data['date'].max() - 7
last_week_popular = train_data[train_data['date'] >= LAST_WEEK_DATE]['item_id'].value_counts().head(100).index.tolist()

for user_id in tqdm(sample_submission['user_id'].unique()):
    candidates.setdefault(user_id, set())
    candidates[user_id].update(last_week_popular)

# --- Генератор 3: Глобально популярные товары ---
print("Генерация кандидатов на основе глобальной популярности...")
globally_popular = train_data['item_id'].value_counts().head(100).index.tolist()

for user_id in tqdm(sample_submission['user_id'].unique()):
    candidates.setdefault(user_id, set())
    candidates[user_id].update(globally_popular)


# --- 3. Этап 2: Ранжирование ---
print("Подготовка данных для ранжирования...")

# Создаем датафрейм с парами (user, candidate_item)
ranker_train_data = []
# Готовим "правильные" ответы для валидации
val_user_items = val_data.groupby('user_id')['item_id'].apply(list).to_dict()

for user_id, user_candidates in tqdm(candidates.items()):
    # Пропускаем пользователей, которых нет в валидационной выборке
    if user_id not in val_user_items:
        continue
    
    true_items = set(val_user_items[user_id])
    for item_id in user_candidates:
        is_target = 1 if item_id in true_items else 0
        ranker_train_data.append((user_id, item_id, is_target))

ranker_df = pd.DataFrame(ranker_train_data, columns=['user_id', 'item_id', 'target'])

# --- Создание признаков для ранжировщика ---
# Признак 1: Глобальная популярность товара
item_popularity = train_data['item_id'].value_counts().reset_index()
item_popularity.columns = ['item_id', 'item_pop_global']
ranker_df = ranker_df.merge(item_popularity, on='item_id', how='left').fillna(0)

# Признак 2: Активность пользователя
user_activity = train_data['user_id'].value_counts().reset_index()
user_activity.columns = ['user_id', 'user_act_global']
ranker_df = ranker_df.merge(user_activity, on='user_id', how='left').fillna(0)

# Признак 3: Количество уникальных товаров у пользователя
user_unique_items = train_data.groupby('user_id')['item_id'].nunique().reset_index()
user_unique_items.columns = ['user_id', 'user_unique_items']
ranker_df = ranker_df.merge(user_unique_items, on='user_id', how='left').fillna(0)

print(f"Размер датасета для ранжировщика: {ranker_df.shape}")
print("Пример данных для ранжировщика:")
print(ranker_df.head())


# --- Обучение LGBMRanker ---
print("Обучение LGBMRanker...")
# Группировка данных по пользователям, это ключевой момент для моделей ранжирования
# Модель будет учиться ранжировать товары внутри каждой группы (для каждого пользователя)
group_sizes = ranker_df.groupby('user_id').size().to_numpy()

features = ['item_pop_global', 'user_act_global', 'user_unique_items']
X_train = ranker_df[features]
y_train = ranker_df['target']

lgb_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="map",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    colsample_bytree=0.8,
    subsample=0.8
)

lgb_ranker.fit(
    X_train,
    y_train,
    group=group_sizes,
    verbose=100
)

# --- 4. Предсказание и формирование сабмишна ---
print("Формирование предсказаний...")

# Создаем датафрейм для предсказания со всеми кандидатами
predict_df_list = []
for user_id, items in tqdm(candidates.items()):
    for item_id in items:
        predict_df_list.append((user_id, item_id))

predict_df = pd.DataFrame(predict_df_list, columns=['user_id', 'item_id'])

# Добавляем те же фичи, что и при обучении
predict_df = predict_df.merge(item_popularity, on='item_id', how='left').fillna(0)
predict_df = predict_df.merge(user_activity, on='user_id', how='left').fillna(0)
predict_df = predict_df.merge(user_unique_items, on='user_id', how='left').fillna(0)

# Предсказываем "оценку" релевантности для каждой пары (user, item)
predict_df['score'] = lgb_ranker.predict(predict_df[features])

# Сортируем товары по оценке для каждого пользователя и берем топ-20
predict_df = predict_df.sort_values(['user_id', 'score'], ascending=[True, False])
top_20_recs = predict_df.groupby('user_id')['item_id'].apply(lambda x: x.head(20).tolist()).to_dict()

# Формируем файл сабмишна
predictions = []
for user_id in sample_submission['user_id']:
    # Если для пользователя по какой-то причине нет реков, рекомендуем глобально популярные
    recs = top_20_recs.get(user_id, globally_popular[:20])
    predictions.append(' '.join(map(str, recs)))

sample_submission['item_id'] = predictions
sample_submission.to_csv('submission.csv', index=False)

print("Файл submission.csv успешно создан!")
print(sample_submission.head())
