import unittest

import numpy as np

from sem4 import SVDRecommender, singular_value_decomposition


class TestSeminar4(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.score = 0
        self.recommender = SVDRecommender()
        self.user_item_matrix = self.recommender.ui_matrix

    def test1_svd_shapes_and_reconstruction(self):
        X = self.user_item_matrix
        k=100
        U, S, V = singular_value_decomposition(X, k=k)
        self.assertEqual(U.shape[1], k)
        self.assertEqual(S.shape[0], k)
        self.assertEqual(V.shape[0], k)
        X_hat = U @ np.diag(S) @ V
        self.assertEqual(X_hat.shape, X.shape)

        diff = X_hat - X
        mean_diff = np.mean(np.abs(diff))
        self.assertAlmostEqual(mean_diff, 0.004, places=3)
        print("\n" + "=" * 80)
        print("test1_svd_shapes_and_reconstruction passed. Score +4")

    def test2_predict_rating(self):
        rating = self.recommender.predict_rating(1, 47, k=10)
        self.assertIsInstance(rating, float)
        self.assertGreaterEqual(rating, 0.0)
        self.assertLessEqual(rating, 5.0)
        self.assertAlmostEqual(rating, 2.0, delta=0.5)
        print("\n" + "=" * 80)
        print("test2_predict_rating passed. Score +3")

    def test3_predict_items_for_user(self):
        recs = self.recommender.predict_items_for_user(1, k=10, n_recommendations=5)
        self.assertEqual(len(recs), 5)
        self.assertEqual(len(set(recs)), 5)
        self.assertTrue(all(isinstance(i, int) for i in recs))
        self.assertTrue(all(i >= 0 for i in recs))
        print("\n" + "=" * 80)
        print("test3_predict_items_for_user passed. Score +3")

        
if __name__ == "__main__":
    unittest.main()
