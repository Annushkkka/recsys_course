"""
Семинар 4. Матричная факторизация
Цель: Разработать методы матричной факторизации для рекомендательной системы.
"""

import numpy as np

from utils import build_user_item_matrix, id_to_movie


def singular_value_decomposition(X: np.array, k: int) -> tuple:
    """
    Разложение матрицы рейтингов X на U, S, V (SVD) и возвращение
    первых k компонент.

    Алгоритм:
    1) Считаем полное сингулярное разложение:
        X = U_full @ diag(S_full) @ V_full
       где U_full.shape = (n_users, n_features),
             S_full.shape = (n_features,),
             V_full.shape = (n_features, n_items)

    2) Сокращаем до первых k латентных факторов:
        U_k = U_full[:, :k]
        S_k = S_full[:k]
        V_k = V_full[:k, :]

    3) Такое сокращение гарантирует низкоранговое приближение
       X_hat = U_k @ diag(S_k) @ V_k.

    Args:
        X: матрица пользователь-фильм (n_users, n_items)
        k: количество латентных факторов, которое сохранится

    Returns:
        U_k: (n_users, k)
        S_k: (k,), сингулярные значения
        V_k: (k, n_items)
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    if k <= 0:
        raise ValueError("k must be positive")

    U_full, S_full, V_full = np.linalg.svd(X, full_matrices=False)
    k_eff = min(k, S_full.shape[0])
    U = U_full[:, :k_eff]
    S = S_full[:k_eff]
    V = V_full[:k_eff, :]
    return U, S, V


class SVDRecommender:
    """
    Класс для построения рекомендаций на основе матричной факторизации.
    При инициализации строится матрица пользователь-фильм и считается
    полное SVD-разложение, после чего для любого k можно быстро получать
    низкоранговое приближение матрицы рейтингов.
    """

    def __init__(self):
        self.ui_matrix = build_user_item_matrix()
        self.U = None
        self.S = None
        self.V = None
        self._build_factorization()

    def _build_factorization(self):
        max_rank = min(self.ui_matrix.shape)
        self.U, self.S, self.V = singular_value_decomposition(self.ui_matrix, k=max_rank)

    def _reconstruct_matrix(self, k: int) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be positive")

        raise NotImplementedError("Реализуйте восстановление матрицы")

    def predict_rating(self, user_id: int, item_id: int, k: int = 20) -> float:
        """
        Предсказывает рейтинг user_id по фильму item_id методом низкорангового
        приближения SVD.

        Алгоритм:
        1) Берём матрицу user-item, построенную при инициализации класса.
        2) Оставляем первые k латентных факторов и восстанавливаем X_hat.
        3) Предсказание для пары (user_id, item_id) берём из X_hat.
        4) Обрезаем результат в диапазон [0.0, 5.0].
        """
        raise NotImplementedError("Реализуйте предсказание рейтинга")

    def predict_items_for_user(
        self, user_id: int, k: int = 20, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы для пользователя user_id по SVD.

        Алгоритм:
        1) Восстанавливаем приближённую матрицу рейтингов X_hat.
        2) Берём прогнозы для заданного пользователя.
        3) Исключаем фильмы, уже оценённые пользователем.
        4) Сортируем кандидатов по убыванию прогнозного рейтинга.
        5) Возвращаем top-n индексы фильмов.
        """
        raise NotImplementedError("Реализуйте рекомендацию фильмов")


if __name__ == "__main__":
    recommender = SVDRecommender()
    k = 100
    U, S, V = singular_value_decomposition(recommender.ui_matrix, k=k)
    print("SVD shapes:", U.shape, S.shape, V.shape)

    X_hat = U @ np.diag(S) @ V
    diff = X_hat - recommender.ui_matrix
    d_min = np.min(diff)
    d_max = np.max(diff)
    d_norm = np.abs(diff).mean()
    print(
        f"Reconstruction error (k={k}): min={d_min:.4f}, max={d_max:.4f}, mean={d_norm:.4f}"
    )

    recs = recommender.predict_items_for_user(1, k=10, n_recommendations=5)
    for rec in recs:
        pred_rating = recommender.predict_rating(1, rec, k=10)
        movie = id_to_movie(rec)
        print(f"Recommend {movie}, predicted rating: {pred_rating:.2f}")
