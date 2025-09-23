"""
Tests for monitoring dashboard module
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
    from monitoring_dashboard import DashboardDataManager, PredictiveAnalyticsEngine, DashboardUI
except ImportError:
    # Skip tests if dependencies not available
    pytest.skip("Dashboard dependencies not available", allow_module_level=True)


class TestDashboardDataManager:
    """Test suite for DashboardDataManager"""

    @pytest.fixture
    def data_manager(self):
        """Create DashboardDataManager instance"""
        return DashboardDataManager()

    def test_data_manager_initialization(self, data_manager):
        """Test data manager initialization"""
        assert isinstance(data_manager.data_cache, dict)
        assert isinstance(data_manager.last_update, dict)

    def test_create_demo_data(self, data_manager):
        """Test demo data creation"""
        demo_data = data_manager._create_demo_data()

        assert isinstance(demo_data, pd.DataFrame)
        assert len(demo_data) > 0
        assert 'trial_id' in demo_data.columns
        assert 'condition' in demo_data.columns
        assert 'endpoint_value' in demo_data.columns

    def test_get_real_time_metrics(self, data_manager):
        """Test real-time metrics calculation"""
        metrics = data_manager.get_real_time_metrics()

        assert isinstance(metrics, dict)
        assert 'total_trials' in metrics
        assert 'active_trials' in metrics
        assert 'total_patients' in metrics
        assert isinstance(metrics['total_trials'], int)
        assert metrics['total_trials'] >= 0


class TestPredictiveAnalyticsEngine:
    """Test suite for PredictiveAnalyticsEngine"""

    @pytest.fixture
    def analytics_engine(self):
        """Create PredictiveAnalyticsEngine instance"""
        return PredictiveAnalyticsEngine()

    @pytest.fixture
    def sample_data(self):
        """Create sample clinical data"""
        np.random.seed(42)
        n = 50

        data = pd.DataFrame({
            'trial_id': [f'NCT{i:07d}' for i in range(n)],
            'condition': np.random.choice(['Epilepsy', 'Diabetes'], n),
            'intervention': np.random.choice(['MSC_autologous', 'MSC_allogeneic'], n),
            'n_patients': np.random.randint(10, 100, n),
            'treatment_group': np.random.randint(5, 50, n),
            'control_group': np.random.randint(5, 50, n),
            'endpoint_value': np.random.normal(50, 15, n),
            'baseline_value': np.random.normal(30, 8, n),
            'follow_up_months': np.random.randint(6, 24, n),
            'phase': np.random.choice(['Phase_1', 'Phase_2'], n),
            'status': np.random.choice(['Completed', 'Ongoing'], n),
            'safety_events': np.random.randint(0, 5, n),
            'country': np.random.choice(['USA', 'Canada'], n)
        })

        return data

    def test_analytics_engine_initialization(self, analytics_engine):
        """Test analytics engine initialization"""
        assert analytics_engine.models_loaded == False
        assert isinstance(analytics_engine.predictors, dict)

    def test_predict_outcome_no_models(self, analytics_engine):
        """Test prediction when models not loaded"""
        patient_features = {
            'n_patients': 50,
            'treatment_group': 25,
            'control_group': 25
        }

        result = analytics_engine.predict_outcome(patient_features)
        assert 'error' in result

    def test_simulate_trial_outcome_basic(self, analytics_engine):
        """Test basic trial outcome simulation"""
        trial_config = {
            'n_treatment': 25,
            'n_control': 25
        }

        result = analytics_engine.simulate_trial_outcome(trial_config)
        assert isinstance(result, dict)
        # Should either have simulation results or error
        assert 'error' in result or 'treatment_group' in result


class TestDashboardUI:
    """Test suite for DashboardUI"""

    @pytest.fixture
    def dashboard_ui(self):
        """Create DashboardUI instance"""
        return DashboardUI()

    def test_dashboard_ui_initialization(self, dashboard_ui):
        """Test dashboard UI initialization"""
        assert isinstance(dashboard_ui.data_manager, DashboardDataManager)
        assert isinstance(dashboard_ui.analytics_engine, PredictiveAnalyticsEngine)

    def test_apply_filters_basic(self, dashboard_ui):
        """Test basic filter application"""
        # Create sample data
        data = pd.DataFrame({
            'condition': ['Epilepsy', 'Diabetes', 'Epilepsy'],
            'phase': ['Phase_1', 'Phase_2', 'Phase_1'],
            'status': ['Completed', 'Ongoing', 'Completed'],
            'endpoint_value': [80, 70, 85]
        })

        filters = {
            'condition': 'Epilepsy',
            'phase': 'All',
            'status': 'All'
        }

        filtered_data = dashboard_ui._apply_filters(data, filters)

        assert len(filtered_data) == 2  # Only epilepsy trials
        assert all(filtered_data['condition'] == 'Epilepsy')

    def test_apply_filters_multiple(self, dashboard_ui):
        """Test multiple filters"""
        data = pd.DataFrame({
            'condition': ['Epilepsy', 'Diabetes', 'Epilepsy', 'Epilepsy'],
            'phase': ['Phase_1', 'Phase_2', 'Phase_1', 'Phase_2'],
            'status': ['Completed', 'Ongoing', 'Completed', 'Ongoing'],
            'endpoint_value': [80, 70, 85, 75]
        })

        filters = {
            'condition': 'Epilepsy',
            'phase': 'Phase_1',
            'status': 'Completed'
        }

        filtered_data = dashboard_ui._apply_filters(data, filters)

        assert len(filtered_data) == 2  # Epilepsy + Phase_1 + Completed
        assert all(filtered_data['condition'] == 'Epilepsy')
        assert all(filtered_data['phase'] == 'Phase_1')
        assert all(filtered_data['status'] == 'Completed')


def test_basic_imports():
    """Test basic imports for dashboard functionality"""
    try:
        import pandas as pd
        import numpy as np
        assert True
    except ImportError as e:
        pytest.fail(f"Required dependency not available: {e}")


def test_data_processing():
    """Test basic data processing functionality"""
    # Test DataFrame operations that dashboard relies on
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'a', 'b', 'a'],
        'C': [10, 20, 30, 40, 50]
    })

    # Test filtering
    filtered = data[data['B'] == 'a']
    assert len(filtered) == 3

    # Test grouping
    grouped = data.groupby('B')['C'].mean()
    assert len(grouped) == 2

    # Test aggregation
    total = data['C'].sum()
    assert total == 150


def test_metrics_calculation():
    """Test metrics calculation functionality"""
    # Simulate metrics calculation like in dashboard
    data = pd.DataFrame({
        'status': ['Completed', 'Ongoing', 'Completed', 'Ongoing', 'Completed'],
        'safety_events': [0, 1, 2, 0, 1],
        'n_patients': [50, 30, 40, 60, 35],
        'p_value': [0.01, 0.05, 0.001, 0.08, 0.02]
    })

    # Test metrics
    total_trials = len(data)
    active_trials = len(data[data['status'] == 'Ongoing'])
    total_patients = data['n_patients'].sum()
    success_rate = len(data[data['p_value'] < 0.05]) / len(data) * 100

    assert total_trials == 5
    assert active_trials == 2
    assert total_patients == 215
    assert success_rate == 80.0  # 4 out of 5 trials have p < 0.05


if __name__ == "__main__":
    pytest.main([__file__])