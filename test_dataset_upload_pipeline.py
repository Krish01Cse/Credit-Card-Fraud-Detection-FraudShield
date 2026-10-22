"""
test_dataset_upload_pipeline.py — Comprehensive Unit Test Suite
Validates all 12 testing requirements specified in Prompt Section 9:
 1. Standard Kaggle creditcard.csv file.
 2. Column names with different capitalization and whitespace (' amount ', 'CLASS', 'time_stamp').
 3. Empty rows and completely empty columns.
 4. Duplicate rows.
 5. Missing values in noncritical columns.
 6. Missing or invalid fraud labels (ensuring missing target rows are NEVER filled with 0).
 7. Renamed columns and ambiguous mappings.
 8. Extra irrelevant columns (merchant, category, cc_num).
 9. A dataset with no Class column (Scenario F: inference only).
 10. A completely invalid or corrupted CSV file.
 11. A valid dataset with a feature schema incompatible with the existing model.
 12. Regression test confirming original missing-column error is handled properly.
"""

import io
import unittest
import numpy as np
import pandas as pd
import backend


class TestDatasetUploadPipeline(unittest.TestCase):
    def setUp(self):
        # Generate base standard Kaggle sample
        self.base_df = backend.generate_sample_dataset(n_samples=50, random_state=42)

    # -----------------------------------------------------------------------
    # 1. Standard Kaggle creditcard.csv
    # -----------------------------------------------------------------------
    def test_01_standard_kaggle_creditcard_csv(self):
        res = backend.ingest_and_validate_dataset(self.base_df)
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "ready")
        self.assertTrue(res["has_target"])
        self.assertTrue(res["can_train"])
        self.assertTrue(res["can_infer"])
        self.assertTrue(res["is_model_compatible"])
        self.assertIn("Time", res["df"].columns)
        self.assertIn("Amount", res["df"].columns)
        self.assertIn("Class", res["df"].columns)
        self.assertEqual(len(res["df"].columns), 31)

    # -----------------------------------------------------------------------
    # 2. Column names with different capitalization and whitespace
    # -----------------------------------------------------------------------
    def test_02_capitalization_and_whitespace(self):
        df_funky = pd.DataFrame({
            "  amount  ": [19.99, 50.00, 120.50, 4.25, 99.00] * 3,
            "CLASS": [0, 1, 0, 0, 1] * 3,
            "  time_stamp  ": range(15),
            " V1 ": np.random.randn(15),
            " V2 ": np.random.randn(15),
        })
        res = backend.ingest_and_validate_dataset(df_funky)
        self.assertTrue(res["success"])
        self.assertTrue(res["has_target"])
        self.assertIn("Amount", res["df"].columns)
        self.assertIn("Class", res["df"].columns)
        self.assertIn("Time", res["df"].columns)
        self.assertEqual(res["column_mapping"]["detected_mapping"]["Amount"], "amount")
        self.assertEqual(res["column_mapping"]["detected_mapping"]["Class"], "CLASS")
        self.assertEqual(res["column_mapping"]["detected_mapping"]["Time"], "time_stamp")

    # -----------------------------------------------------------------------
    # 3. Empty rows and completely empty columns
    # -----------------------------------------------------------------------
    def test_03_empty_rows_and_empty_columns(self):
        df_sparse = self.base_df.copy()
        # Add 3 completely blank rows (all NaN)
        blank_row = pd.DataFrame([[np.nan] * len(df_sparse.columns)], columns=df_sparse.columns)
        df_with_blanks = pd.concat([df_sparse, blank_row, blank_row, blank_row], ignore_index=True)
        # Add 2 completely empty columns
        df_with_blanks["totally_empty_col_1"] = np.nan
        df_with_blanks["totally_empty_col_2"] = None

        res = backend.ingest_and_validate_dataset(df_with_blanks)
        self.assertTrue(res["success"])
        self.assertEqual(res["sanitization_summary"]["empty_rows_removed"], 3)
        self.assertEqual(res["sanitization_summary"]["empty_cols_removed"], 2)
        self.assertEqual(len(res["df"]), len(self.base_df))

    # -----------------------------------------------------------------------
    # 4. Duplicate rows
    # -----------------------------------------------------------------------
    def test_04_duplicate_rows(self):
        dups = self.base_df.iloc[[0, 1, 2]]
        df_with_dups = pd.concat([self.base_df, dups], ignore_index=True)
        res = backend.ingest_and_validate_dataset(df_with_dups)
        self.assertTrue(res["success"])
        self.assertEqual(res["sanitization_summary"]["duplicate_rows_removed"], 3)
        self.assertEqual(len(res["df"]), len(self.base_df))

    # -----------------------------------------------------------------------
    # 5. Missing values in noncritical columns
    # -----------------------------------------------------------------------
    def test_05_missing_values_in_noncritical_columns(self):
        df_missing = self.base_df.copy()
        df_missing.loc[0, "Amount"] = np.nan
        df_missing.loc[1, "V1"] = np.nan
        df_missing.loc[2, "V2"] = np.nan
        res = backend.ingest_and_validate_dataset(df_missing, imputation_strategy="median")
        self.assertTrue(res["success"])
        # Verify valid rows were NOT deleted
        self.assertEqual(len(res["df"]), len(self.base_df))
        self.assertEqual(res["df"]["Amount"].isnull().sum(), 0)
        self.assertEqual(res["df"]["V1"].isnull().sum(), 0)

    # -----------------------------------------------------------------------
    # 6. Missing or invalid fraud labels (NEVER fill target with 0)
    # -----------------------------------------------------------------------
    def test_06_missing_or_invalid_fraud_labels_safeguard(self):
        df_label_leak = self.base_df.copy()
        # Set 4 target rows to NaN / invalid text
        df_label_leak.loc[0, "Class"] = np.nan
        df_label_leak.loc[1, "Class"] = None
        df_label_leak.loc[2, "Class"] = "UNKNOWN_LABEL"
        df_label_leak.loc[3, "Class"] = np.nan

        res = backend.ingest_and_validate_dataset(df_label_leak)
        self.assertTrue(res["success"])
        # The 4 invalid target rows must be dropped from training set, NOT filled with 0
        self.assertEqual(len(res["df"]), len(self.base_df) - 4)
        # Target must only contain {0, 1}
        self.assertEqual(set(res["df"]["Class"].unique()), {0, 1})

    # -----------------------------------------------------------------------
    # 7. Renamed columns and ambiguous mappings
    # -----------------------------------------------------------------------
    def test_07_renamed_columns_and_ambiguous_mappings(self):
        df_renamed = pd.DataFrame({
            "amt": [15.5, 45.0, 120.0, 9.99, 88.0] * 4,
            "unix_time": range(20),
            "is_fraud": [0, 1, 0, 0, 1] * 4,
            "metric_1": np.random.randn(20),
            "metric_2": np.random.randn(20),
        })
        res = backend.ingest_and_validate_dataset(df_renamed)
        self.assertTrue(res["success"])
        self.assertEqual(res["column_mapping"]["detected_mapping"]["Amount"], "amt")
        self.assertEqual(res["column_mapping"]["detected_mapping"]["Class"], "is_fraud")
        self.assertEqual(res["column_mapping"]["detected_mapping"]["Time"], "unix_time")

        # Ambiguous test: multiple amounts
        df_ambig = pd.DataFrame({
            "amt": [10, 20] * 5,
            "charge_amount": [10, 20] * 5,
            "is_fraud": [0, 1] * 5,
        })
        res_ambig = backend.ingest_and_validate_dataset(df_ambig)
        self.assertTrue(res_ambig.get("needs_mapping") or res_ambig.get("success"))

    # -----------------------------------------------------------------------
    # 8. Extra irrelevant columns
    # -----------------------------------------------------------------------
    def test_08_extra_irrelevant_columns_excluded_from_model_features(self):
        df_extra = self.base_df.copy()
        df_extra["merchant"] = "Target Store #102"
        df_extra["category"] = "Groceries"
        df_extra["cc_num"] = "4000123456789010"
        df_extra["zip"] = "78701"

        res = backend.ingest_and_validate_dataset(df_extra)
        self.assertTrue(res["success"])
        # Unmapped columns tracked
        self.assertIn("merchant", res["column_mapping"]["unmapped_columns"])
        self.assertIn("category", res["column_mapping"]["unmapped_columns"])

        # When preprocessed for model, feature columns must strictly be the 30 standard features
        X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(res["df"])
        self.assertEqual(len(f_names), 30)
        self.assertNotIn("merchant", f_names)
        self.assertNotIn("category", f_names)
        self.assertEqual(X_tr.shape[1], 30)

    # -----------------------------------------------------------------------
    # 9. Dataset with no Class column (Scenario F: inference only)
    # -----------------------------------------------------------------------
    def test_09_unlabeled_dataset_no_class_column(self):
        df_unlabeled = self.base_df.drop(columns=["Class"])
        res = backend.ingest_and_validate_dataset(df_unlabeled)
        self.assertTrue(res["success"])
        self.assertFalse(res["has_target"])
        self.assertFalse(res["can_train"])
        self.assertTrue(res["can_infer"])
        self.assertEqual(res["status"], "unlabeled_inference_only")
        self.assertIn("no fraud label found", res["message"])

        # Batch inference must run cleanly on this unlabeled dataset
        model = backend.create_classifier("Decision Tree")
        # Train on base dataset first
        X_tr, _, y_tr, _, scl, _ = backend.preprocess_data(self.base_df)
        model.fit(X_tr, y_tr)
        batch_res = backend.batch_predict_dataset(model, scl, res["df"])
        self.assertEqual(batch_res["total_scanned"], len(df_unlabeled))
        self.assertIn("predicted_fraud_count", batch_res)
        self.assertIn("flagged_transactions", batch_res)

    # -----------------------------------------------------------------------
    # 10. Completely invalid or corrupted CSV file
    # -----------------------------------------------------------------------
    def test_10_completely_invalid_or_corrupted_csv(self):
        empty_bytes = b""
        _, err = backend.parse_csv_bytes(empty_bytes)
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())

        invalid_df = pd.DataFrame()
        res = backend.ingest_and_validate_dataset(invalid_df)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "incompatible")

    # -----------------------------------------------------------------------
    # 11. Valid dataset with incompatible feature schema
    # -----------------------------------------------------------------------
    def test_11_incompatible_feature_schema_handled(self):
        # Dataset with only categorical columns and no numeric features or amounts
        df_incompat = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "city": ["New York", "London", "Tokyo"],
            "status": ["active", "pending", "closed"],
        })
        res = backend.ingest_and_validate_dataset(df_incompat)
        self.assertFalse(res["can_train"])

    # -----------------------------------------------------------------------
    # 12. Regression test confirming original missing-column error handled gracefully
    # -----------------------------------------------------------------------
    def test_12_regression_missing_column_error_handled_gracefully(self):
        # A dataset like fraudTest.csv with 'amt' and 'is_fraud'
        df_external = pd.DataFrame({
            "amt": [12.50, 250.00, 4.99, 1200.00, 35.00] * 3,
            "is_fraud": [0, 1, 0, 1, 0] * 3,
            "trans_date_trans_time": [f"2026-01-01 10:{i:02d}:00" for i in range(15)],
            "lat": [34.0, 40.7, 51.5, 48.8, 35.6] * 3,
            "long": [-118.2, -74.0, -0.1, 2.3, 139.6] * 3,
        })
        # Formerly this threw 400 Missing required column(s): Amount, Class, Time
        adapted, err = backend.validate_dataframe(df_external)
        self.assertIsNone(err)
        self.assertIsNotNone(adapted)
        self.assertIn("Amount", adapted.columns)
        self.assertIn("Class", adapted.columns)
        self.assertIn("Time", adapted.columns)
        self.assertEqual(len(adapted), 15)


if __name__ == "__main__":
    unittest.main()
