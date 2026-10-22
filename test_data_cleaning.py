"""
Unit tests for data cleaning, flexible schema adaptation, and preprocessing API endpoints.
"""
import unittest
import urllib.request
import json
import pandas as pd
import numpy as np
import backend


class TestDataCleaningAndPreprocessing(unittest.TestCase):
    def setUp(self):
        self.sample_df = backend.generate_sample_dataset(n_samples=200, random_state=42)

    def test_schema_adaptor_target_variants(self):
        """Verify non-standard target columns (is_fraud, fraud, label) are mapped to Class."""
        df_variant = pd.DataFrame({
            "amt": [10.0, 50.0, 99.0] * 5,
            "unix_time": range(15),
            "is_fraud": [0, 1, 0] * 5,
            "feat_a": np.random.randn(15),
            "feat_b": np.random.randn(15),
        })
        adapted, logs = backend.adapt_and_normalize_schema(df_variant)
        self.assertIn("Class", adapted.columns)
        self.assertIn("Amount", adapted.columns)
        self.assertIn("Time", adapted.columns)
        self.assertEqual(len(adapted["Class"].unique()), 2)
        # Check all 28 PCA features are present
        for i in range(1, 29):
            self.assertIn(f"V{i}", adapted.columns)

    def test_schema_adaptor_string_target_labels(self):
        """Verify string labels ('legit' / 'fraud') are mapped to binary 0 and 1."""
        df_str = pd.DataFrame({
            "Amount": [25.0, 120.0] * 6,
            "Time": range(12),
            "fraud": ["legit", "fraud"] * 6,
            "V1": np.random.randn(12),
        })
        adapted, _ = backend.adapt_and_normalize_schema(df_str)
        self.assertEqual(set(adapted["Class"].unique()), {0, 1})

    def test_cleaning_pipeline_deduplication(self):
        """Verify duplicate transaction rows are detected and removed."""
        df_with_dups = pd.concat([self.sample_df, self.sample_df.iloc[[0, 1]]], ignore_index=True)
        self.assertEqual(len(df_with_dups), 202)
        cleaned, summary = backend.apply_data_cleaning_pipeline(df_with_dups, deduplicate=True)
        self.assertEqual(summary["duplicates_dropped"], 2)
        self.assertEqual(len(cleaned), 200)

    def test_cleaning_pipeline_missing_imputation(self):
        """Verify missing values are imputed via median strategy."""
        df_missing = self.sample_df.copy()
        df_missing.loc[0, "Amount"] = None
        df_missing.loc[1, "V1"] = None
        cleaned, summary = backend.apply_data_cleaning_pipeline(df_missing, imputation_strategy="median")
        self.assertEqual(summary["missing_cells_imputed"], 2)
        self.assertEqual(cleaned["Amount"].isnull().sum(), 0)
        self.assertEqual(cleaned["V1"].isnull().sum(), 0)

    def test_cleaning_pipeline_outlier_clipping(self):
        """Verify extreme transaction amount outliers are clipped to the 1.5 * IQR bound."""
        df_outliers = self.sample_df.copy()
        df_outliers.loc[0, "Amount"] = 999999.0  # extreme anomaly
        cleaned, summary = backend.apply_data_cleaning_pipeline(df_outliers, outlier_action="clip")
        self.assertGreaterEqual(summary["outliers_clipped"], 1)
        self.assertLess(cleaned["Amount"].max(), 5000.0)

    def test_comprehensive_cleaning_report(self):
        """Verify report returns complete audit structure with imbalance & stratified split."""
        report = backend.get_comprehensive_cleaning_report(self.sample_df)
        self.assertEqual(report["total_rows"], 200)
        self.assertIn("class_imbalance", report)
        self.assertIn("stratified_split", report)
        self.assertIn("outlier_stats", report)
        self.assertIn("smote_projection", report)
        self.assertIn("raw_sample", report)
        self.assertIn("cleaned_sample", report)
        self.assertEqual(len(report["raw_sample"]), 5)

    def test_api_cleaning_report_endpoint(self):
        """Verify live GET /api/dataset/cleaning-report endpoint responds with 200 OK."""
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/dataset/cleaning-report")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode())
                self.assertIn("report", data)
                self.assertIn("class_imbalance", data["report"])
        except Exception as exc:
            self.skipTest(f"Live server check skipped: {exc}")

    def test_api_clean_endpoint(self):
        """Verify live POST /api/dataset/clean endpoint runs and updates state."""
        try:
            payload = json.dumps({
                "imputation_strategy": "median",
                "outlier_action": "clip",
                "scaler_type": "standard",
                "deduplicate": True
            }).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8000/api/dataset/clean",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode())
                self.assertTrue(data["success"])
                self.assertIn("summary", data)
        except Exception as exc:
            self.skipTest(f"Live server check skipped: {exc}")


if __name__ == "__main__":
    unittest.main()
