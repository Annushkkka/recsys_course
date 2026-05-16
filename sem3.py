"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)
        genres_matrix = vectorizer.fit_transform(self.movies_df["genres"])

        max_movie_id = self.movies_df["movieId"].max()
        n_genres = genres_matrix.shape[1]

        self.embeddings = np.zeros((max_movie_id + 1, n_genres))

        for i, row in self.movies_df.iterrows():
            movie_id = int(row["movieId"])
            self.embeddings[movie_id] = genres_matrix[i].toarray()[0]
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        # 1) Берём вектор целевого фильма: target_vec.
        target_vec = self.embeddings[item_id]

        if np.linalg.norm(target_vec) == 0:
            return 0.0

        # 2) Находим все фильмы, оцененные пользователем.
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return 0.0

        rated_vectors = self.embeddings[rated_items]
        rated_scores = user_ratings[rated_items]

        # 3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        target_norm = np.linalg.norm(target_vec)
        rated_norms = np.linalg.norm(rated_vectors, axis=1)

        valid_mask = rated_norms > 0

        if not np.any(valid_mask):
            return 0.0

        rated_items = rated_items[valid_mask]
        rated_vectors = rated_vectors[valid_mask]
        rated_scores = rated_scores[valid_mask]
        rated_norms = rated_norms[valid_mask]

        similarities = rated_vectors @ target_vec / (rated_norms * target_norm)

        # 4) Отбираем топ-k похожих оцененных фильмов.
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        top_ratings = rated_scores[top_indices]

        # 6) Если не удаётся предсказать, возвращаем 0.0.
        sum_sim = top_similarities.sum()
        if sum_sim == 0:
            return 0.0

        # 5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        pred = np.dot(top_similarities, top_ratings) / sum_sim

        # 7) Клипируем результат в [0.0, 5.0].
        return float(np.clip(pred, 0.0, 5.0))

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        # 1) Берем все фильмы, которые оценил пользователь.
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return []

        rated_vectors = self.embeddings[rated_items]
        rated_scores = user_ratings[rated_items]

        # 3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        user_profile = np.average(rated_vectors, axis=0, weights=rated_scores)

        if np.linalg.norm(user_profile) == 0:
            return []

        # 4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        user_rated = user_ratings > 0
        candidate_items = np.where(~user_rated)[0]

        candidate_vectors = self.embeddings[candidate_items]

        profile_norm = np.linalg.norm(user_profile)
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)

        valid_mask = candidate_norms > 0
        candidate_items = candidate_items[valid_mask]
        candidate_vectors = candidate_vectors[valid_mask]
        candidate_norms = candidate_norms[valid_mask]

        similarities = candidate_vectors @ user_profile / (candidate_norms * profile_norm)

        # 5) Сортируем по убыванию сходства и возвращаем top-n.
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        recommendations = [int(candidate_items[i]) for i in top_indices]

        return recommendations


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
