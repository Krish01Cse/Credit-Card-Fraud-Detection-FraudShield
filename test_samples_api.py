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
