import unittest

from sem3 import ContentRecommender
from utils import build_user_item_matrix


class TestSeminar3(unittest.TestCase):
    def setUp(self):
        self.score = 0
        self.X = build_user_item_matrix()
        self.recommender = ContentRecommender()
        return super().setUp()

    def test1_vectorize_content(self):
        # Get the embedding vector for movie with ID 1
        vec = self.recommender.embeddings[1]
        self.assertEqual(vec.ndim, 1)
        self.assertGreaterEqual(vec.sum(), 0)
        print("\n" + "=" * 80)
        print("test1_vectorize_content passed. Score +4")

    def test2_predict_rating_bounds(self):
        rating = self.recommender.predict_rating(1, 47, k=5)
        self.assertIsInstance(rating, float)
        self.assertGreaterEqual(rating, 0.0)
        self.assertLessEqual(rating, 5.0)
        print("\n" + "=" * 80)
        print("test2_predict_rating_bounds passed. Score +3")

    def test3_predict_items_for_user(self):
        recs = self.recommender.predict_items_for_user(1, k=5, n_recommendations=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(len(set(recs)), 5)
        self.assertTrue(all(isinstance(i, int) for i in recs))
        self.assertTrue(all(0 <= i < self.X.shape[1] for i in recs))
        print("\n" + "=" * 80)
        print("test3_predict_items_for_user passed. Score +3")


if __name__ == "__main__":
    unittest.main()
