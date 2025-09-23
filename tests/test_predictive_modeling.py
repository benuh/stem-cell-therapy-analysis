"""
Tests for predictive modeling module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from predictive_modeling import TreatmentOutcomePredictor, TreatmentOptimizer
except ImportError:
    # Skip tests if dependencies not available
    pytest.skip("Predictive modeling dependencies not available", allow_module_level=True)


class TestTreatmentOutcomePredictor:
    """Test suite for TreatmentOutcomePredictor"""

    @pytest.fixture
    def sample_data(self):
        """Create sample clinical trial data for testing"""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'trial_id': [f'NCT{i:07d}' for i in range(n)],
            'condition': np.random.choice(['Epilepsy', 'Diabetes', 'Heart_Failure'], n),
            'intervention': np.random.choice(['MSC_autologous', 'MSC_allogeneic'], n),
            'n_patients': np.random.randint(10, 200, n),
            'treatment_group': np.random.randint(5, 100, n),
            'control_group': np.random.randint(5, 100, n),
            'endpoint_value': np.random.normal(50, 20, n),
            'baseline_value': np.random.normal(30, 10, n),
            'follow_up_months': np.random.randint(6, 36, n),
            'phase': np.random.choice(['Phase_1', 'Phase_2', 'Phase_3'], n),
            'status': np.random.choice(['Completed', 'Ongoing'], n),
            'safety_events': np.random.randint(0, 10, n),
            'country': np.random.choice(['USA', 'Canada', 'UK'], n)
        })

        return data

    @pytest.fixture
    def predictor(self):
        """Create predictor instance"""
        return TreatmentOutcomePredictor()

    def test_predictor_initialization(self, predictor):
        """Test predictor initialization"""
        assert predictor.target_variable == 'endpoint_value'
        assert isinstance(predictor.models, dict)
        assert isinstance(predictor.trained_models, dict)

    def test_initialize_models(self, predictor):
        """Test model initialization"""
        predictor.initialize_models()

        # Check that models are initialized
        assert len(predictor.models) > 0
        assert 'random_forest' in predictor.models
        assert 'xgboost' in predictor.models

        # Check ensemble models are initialized
        assert len(predictor.ensemble_models) > 0
        assert 'voting_regressor' in predictor.ensemble_models

    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation"""
        X, y, feature_names = predictor.prepare_features(sample_data)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(feature_names, list)
        assert X.shape[0] == len(sample_data)
        assert y.shape[0] == len(sample_data)

    def test_feature_engineering(self, predictor, sample_data):
        """Test feature engineering"""
        # Get feature columns
        exclude_cols = [predictor.target_variable, 'trial_id']
        feature_cols = [col for col in sample_data.columns if col not in exclude_cols]
        X = sample_data[feature_cols].copy()

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes

        X_engineered = predictor._engineer_features(X)

        # Check that engineered features are added
        assert X_engineered.shape[1] >= X.shape[1]
        assert isinstance(X_engineered, pd.DataFrame)

    @patch('predictive_modeling.RandomForestRegressor')
    @patch('predictive_modeling.xgb.XGBRegressor')
    def test_train_all_models_basic(self, mock_xgb, mock_rf, predictor, sample_data):
        """Test basic model training functionality"""
        # Mock the models to avoid actual training
        mock_rf_instance = Mock()
        mock_rf_instance.fit.return_value = None
        mock_rf_instance.predict.return_value = np.random.random(50)
        mock_rf_instance.feature_importances_ = np.random.random(10)
        mock_rf.return_value = mock_rf_instance

        mock_xgb_instance = Mock()
        mock_xgb_instance.fit.return_value = None
        mock_xgb_instance.predict.return_value = np.random.random(50)
        mock_xgb_instance.feature_importances_ = np.random.random(10)
        mock_xgb.return_value = mock_xgb_instance

        # Test training
        results = predictor.train_all_models(sample_data, optimize_hyperparams=False)

        assert isinstance(results, dict)
        assert 'data_info' in results

    def test_predict_treatment_outcome_no_models(self, predictor):
        """Test prediction when no models are trained"""
        patient_features = {
            'n_patients': 50,
            'treatment_group': 25,
            'control_group': 25,
            'baseline_value': 10,
            'follow_up_months': 12
        }

        result = predictor.predict_treatment_outcome(patient_features)
        assert 'error' in result

    def test_patient_features_to_vector(self, predictor):
        """Test conversion of patient features to vector"""
        patient_features = {
            'n_patients': 50,
            'treatment_group': 25,
            'control_group': 25,
            'baseline_value': 10,
            'follow_up_months': 12,
            'safety_events': 1
        }

        vector = predictor._patient_features_to_vector(patient_features)
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0


class TestTreatmentOptimizer:
    """Test suite for TreatmentOptimizer"""

    @pytest.fixture
    def predictor(self):
        """Create mock predictor"""
        predictor = Mock()
        predictor.trained_models = {'random_forest': Mock()}
        predictor.predict_treatment_outcome.return_value = {
            'predicted_outcome': 75.0,
            'model_used': 'random_forest'
        }
        return predictor

    @pytest.fixture
    def optimizer(self, predictor):
        """Create optimizer instance"""
        return TreatmentOptimizer(predictor)

    def test_optimizer_initialization(self, optimizer, predictor):
        """Test optimizer initialization"""
        assert optimizer.predictor == predictor
        assert isinstance(optimizer.optimization_results, dict)

    def test_optimize_treatment_protocol_no_models(self):
        """Test optimization when no models are available"""
        predictor = Mock()
        predictor.trained_models = {}

        optimizer = TreatmentOptimizer(predictor)

        base_patient = {'age': 55}
        variable_params = ['follow_up_months']
        param_ranges = {'follow_up_months': (6, 24)}

        result = optimizer.optimize_treatment_protocol(
            base_patient, variable_params, param_ranges
        )

        assert 'error' in result

    def test_optimize_treatment_protocol_basic(self, optimizer):
        """Test basic optimization functionality"""
        base_patient = {
            'age': 55,
            'condition': 'Epilepsy'
        }
        variable_params = ['follow_up_months']
        param_ranges = {'follow_up_months': (6, 24)}

        result = optimizer.optimize_treatment_protocol(
            base_patient, variable_params, param_ranges
        )

        # Should return optimization results
        assert isinstance(result, dict)


def test_imports():
    """Test that all required modules can be imported"""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        assert True
    except ImportError as e:
        pytest.fail(f"Required dependency not available: {e}")


def test_data_types():
    """Test basic data type handling"""
    # Test numpy array creation
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.dtype in [np.int64, np.int32]

    # Test pandas DataFrame creation
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    assert df.shape == (3, 2)


if __name__ == "__main__":
    pytest.main([__file__])