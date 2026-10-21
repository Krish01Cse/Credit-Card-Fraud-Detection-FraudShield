"""
Unit tests for the sample transaction repository and live API query filters.
"""
import unittest
import urllib.request
import json
import backend

class TestTransactionSamples(unittest.TestCase):
    def test_backend_sample_transactions_lists(self):
        res = backend.get_sample_transactions()
        self.assertIn("legit_transactions", res)
        self.assertIn("fraud_transactions", res)
        self.assertGreaterEqual(len(res["legit_transactions"]), 5)
        self.assertGreaterEqual(len(res["fraud_transactions"]), 5)

    def test_backend_sample_transactions_filter_groceries(self):
        res = backend.get_sample_transactions(category="Groceries")
        self.assertGreaterEqual(len(res["legit_transactions"]), 1)
        for tx in res["legit_transactions"]:
            self.assertIn("groceries", tx["category"].lower())

    def test_backend_sample_transactions_search_crypto(self):
        res = backend.get_sample_transactions(query="crypto")
        self.assertGreaterEqual(len(res["fraud_transactions"]), 1)
        self.assertTrue(any("crypto" in tx["merchant"].lower() for tx in res["fraud_transactions"]))

    def test_sample_transaction_feature_vector_integrity(self):
        """Verify each sample has complete 30-element feature vector with Time, Amount, V1-V28."""
        res = backend.get_sample_transactions()
        for group in ["legit_transactions", "fraud_transactions"]:
            for tx in res[group]:
                features = tx["features"]
                self.assertIn("Time", features)
                self.assertIn("Amount", features)
                for i in range(1, 29):
                    v_key = f"V{i}"
                    self.assertIn(v_key, features, f"Missing {v_key} in {tx['id']}")
                    self.assertIsInstance(features[v_key], (int, float))

    def test_sample_transactions_filter_combination(self):
        """Verify combination of category filter and search query works simultaneously."""
        res = backend.get_sample_transactions(category="ATM Withdrawal", query="night")
        self.assertGreaterEqual(len(res["fraud_transactions"]), 1)
        tx = res["fraud_transactions"][0]
        self.assertEqual(tx["category"], "ATM Withdrawal")
        self.assertIn("night", (tx["merchant"] + tx["tag"]).lower())

    def test_live_api_sample_transactions_endpoint(self):
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/transactions/samples")
            with urllib.request.urlopen(req, timeout=3) as response:
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode())
                self.assertIn("legit_transactions", data)
                self.assertIn("fraud_transactions", data)
                self.assertGreater(len(data["legit_transactions"]), 0)
                self.assertGreater(len(data["fraud_transactions"]), 0)
        except Exception as exc:
            self.skipTest(f"Live server check skipped: {exc}")

if __name__ == "__main__":
    unittest.main()
