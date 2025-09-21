"""
Test suite for statistical models module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistical_models import (
    SurvivalAnalysisModel,
    BayesianMetaAnalysis,
    TreatmentResponsePredictor
)


class TestSurvivalAnalysisModel:
    """Test cases for SurvivalAnalysisModel"""

    def setup_method(self):
        """Set up test fixtures"""
        self.model = SurvivalAnalysisModel()
        self.sample_data = pd.DataFrame({
            'trial_id': ['T1', 'T2', 'T3', 'T4'],
            'follow_up_months': [6, 12, 18, 24],
            'endpoint_value': [50, 75, 80, 60],
            'condition': ['Epilepsy', 'Diabetes', 'Epilepsy', 'Diabetes'],
            'intervention': ['MSC', 'VX-880', 'MSC', 'VX-880'],
            'n_patients': [25, 50, 30, 40]
        })

    def test_prepare_survival_data_valid_input(self):
        """Test survival data preparation with valid input"""
        result = self.model.prepare_survival_data(self.sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
        assert 'duration' in result.columns
        assert 'event' in result.columns

    def test_prepare_survival_data_empty_input(self):
        """Test survival data preparation with empty input"""
        empty_df = pd.DataFrame()
        result = self.model.prepare_survival_data(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_kaplan_meier_analysis_sufficient_data(self):
        """Test Kaplan-Meier analysis with sufficient data"""
        self.model.prepare_survival_data(self.sample_data)

        with patch.object(self.model.kmf, 'fit') as mock_fit:
            mock_fit.return_value = None
            self.model.kmf.median_survival_time_ = 15.0
            self.model.kmf.survival_function_ = pd.DataFrame({'KM_estimate': [1.0, 0.8, 0.6, 0.4]})

            result = self.model.kaplan_meier_analysis()

            assert 'overall' in result
            assert 'median_survival' in result['overall']

    def test_kaplan_meier_analysis_no_data(self):
        """Test Kaplan-Meier analysis with no prepared data"""
        with pytest.raises(ValueError, match="Must prepare survival data first"):
            self.model.kaplan_meier_analysis()


class TestBayesianMetaAnalysis:
    """Test cases for BayesianMetaAnalysis"""

    def setup_method(self):
        """Set up test fixtures"""
        self.model = BayesianMetaAnalysis()
        self.effect_sizes = np.array([0.5, 0.8, 0.3, 0.6])
        self.standard_errors = np.array([0.1, 0.15, 0.08, 0.12])
        self.study_names = ['Study1', 'Study2', 'Study3', 'Study4']

    @patch('statistical_models.pm.sample')
    @patch('statistical_models.pm.Model')
    def test_hierarchical_meta_analysis_valid_input(self, mock_model, mock_sample):
        """Test hierarchical meta-analysis with valid input"""
        # Mock PyMC objects
        mock_trace = Mock()
        mock_trace.posterior = {
            'mu': Mock(values=np.random.normal(0.5, 0.1, (1000, 4))),
            'tau': Mock(values=np.random.gamma(1, 1, (1000, 4))),
            'theta': Mock(values=np.random.normal(0.5, 0.1, (1000, 4, 4)))
        }
        mock_sample.return_value = mock_trace

        with patch('statistical_models.az.hdi') as mock_hdi:
            mock_hdi.return_value = {'mu': np.array([0.3, 0.7]), 'tau': np.array([0.1, 0.3])}

            result = self.model.hierarchical_meta_analysis(
                self.effect_sizes, self.standard_errors, self.study_names
            )

            assert 'population_effect' in result
            assert 'heterogeneity' in result
            assert 'study_effects' in result

    def test_hierarchical_meta_analysis_insufficient_data(self):
        """Test hierarchical meta-analysis with insufficient data"""
        small_effects = np.array([0.5])
        small_errors = np.array([0.1])
        small_names = ['Study1']

        # Should handle small datasets gracefully
        result = self.model.hierarchical_meta_analysis(small_effects, small_errors, small_names)

        # The function should either complete successfully or handle the error gracefully
        assert isinstance(result, dict)

    def test_prediction_interval_no_trace(self):
        """Test prediction interval calculation without fitted model"""
        with pytest.raises(ValueError, match="Must run meta-analysis first"):
            self.model.prediction_interval()


class TestTreatmentResponsePredictor:
    """Test cases for TreatmentResponsePredictor"""

    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = TreatmentResponsePredictor()
        self.sample_data = pd.DataFrame({
            'trial_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'n_patients': [25, 50, 30, 40, 35],
            'endpoint_value': [50, 75, 80, 60, 70],
            'follow_up_months': [6, 12, 18, 24, 15],
            'safety_events': [1, 2, 0, 1, 1],
            'condition': ['Epilepsy', 'Diabetes', 'Epilepsy', 'Diabetes', 'Epilepsy'],
            'intervention': ['MSC', 'VX-880', 'MSC', 'VX-880', 'NRTX'],
            'phase': ['Phase_1', 'Phase_2', 'Phase_1', 'Phase_3', 'Phase_2']
        })

    def test_prepare_features_valid_data(self):
        """Test feature preparation with valid data"""
        X, y, feature_names = self.predictor.prepare_features(self.sample_data)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(X) == len(self.sample_data)
        assert len(y) == len(self.sample_data)
        assert len(X[0]) == len(feature_names)

    def test_prepare_features_missing_values(self):
        """Test feature preparation with missing values"""
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, 'n_patients'] = np.nan
        data_with_nan.loc[1, 'endpoint_value'] = np.nan

        X, y, feature_names = self.predictor.prepare_features(data_with_nan)

        # Should handle NaN values by filling with median
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_train_models_insufficient_data(self):
        """Test model training with insufficient data"""
        small_data = self.sample_data.head(2)  # Only 2 samples

        result = self.predictor.train_models(small_data)

        # Should handle insufficient data gracefully
        assert isinstance(result, dict)

    def test_predict_treatment_response_no_models(self):
        """Test prediction without trained models"""
        patient_features = {
            'n_patients': 50,
            'follow_up_months': 12,
            'condition': 'Epilepsy'
        }

        result = self.predictor.predict_treatment_response(patient_features)

        assert 'error' in result
        assert 'No trained models' in result['error']

    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        X_input = pd.DataFrame({
            'n_patients': [25, 50, 30],
            'follow_up_months': [6, 12, 18],
            'treatment_group': [15, 25, 20],
            'control_group': [10, 25, 10]
        })

        X_engineered = self.predictor._engineer_features(X_input)

        # Should have more features than input due to engineering
        assert len(X_engineered.columns) > len(X_input.columns)

        # Should contain interaction terms
        interaction_cols = [col for col in X_engineered.columns if '_x_' in col]
        assert len(interaction_cols) > 0


class TestIntegrationScenarios:
    """Integration tests for multiple components working together"""

    def setup_method(self):
        """Set up test fixtures for integration tests"""
        self.sample_data = pd.DataFrame({
            'trial_id': [f'T{i}' for i in range(10)],
            'n_patients': np.random.randint(20, 100, 10),
            'endpoint_value': np.random.uniform(40, 90, 10),
            'follow_up_months': np.random.randint(6, 24, 10),
            'safety_events': np.random.randint(0, 3, 10),
            'condition': np.random.choice(['Epilepsy', 'Diabetes'], 10),
            'intervention': np.random.choice(['MSC', 'VX-880', 'NRTX'], 10)
        })

    def test_end_to_end_survival_analysis(self):
        """Test complete survival analysis workflow"""
        model = SurvivalAnalysisModel()

        # Prepare data
        survival_data = model.prepare_survival_data(self.sample_data)
        assert len(survival_data) > 0

        # Run analysis (with mocked fitting to avoid pymc dependency issues)
        with patch.object(model.kmf, 'fit'):
            model.kmf.median_survival_time_ = 15.0
            model.kmf.survival_function_ = pd.DataFrame({'KM_estimate': [1.0, 0.8, 0.6]})

            result = model.kaplan_meier_analysis()
            assert 'overall' in result

    def test_end_to_end_prediction_workflow(self):
        """Test complete prediction workflow"""
        predictor = TreatmentResponsePredictor()

        # Prepare features
        X, y, feature_names = predictor.prepare_features(self.sample_data)
        assert len(X) > 0
        assert len(y) > 0

        # Training would require mocking or simplified models for unit tests
        # This ensures the data preparation pipeline works correctly


def test_module_imports():
    """Test that all required modules can be imported"""
    try:
        from statistical_models import (
            SurvivalAnalysisModel,
            BayesianMetaAnalysis,
            TreatmentResponsePredictor,
            meta_analysis_diabetes_hba1c,
            survival_analysis_treatment_durability
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")


if __name__ == "__main__":
    pytest.main([__file__])