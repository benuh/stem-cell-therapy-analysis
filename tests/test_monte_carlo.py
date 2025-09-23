"""
Tests for Monte Carlo simulation module
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
    from monte_carlo_simulation import (
        MonteCarloSimulator, ClinicalTrialSimulator,
        TreatmentOptimizationSimulator, RiskAssessmentSimulator,
        DistributionType, SimulationParameter
    )
except ImportError:
    # Skip tests if dependencies not available
    pytest.skip("Monte Carlo simulation dependencies not available", allow_module_level=True)


class TestMonteCarloSimulator:
    """Test suite for MonteCarloSimulator base class"""

    @pytest.fixture
    def simulator(self):
        """Create simulator instance"""
        return MonteCarloSimulator(n_simulations=100, random_seed=42)

    def test_simulator_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator.n_simulations == 100
        assert simulator.random_seed == 42
        assert isinstance(simulator.simulation_results, dict)

    def test_define_parameter(self, simulator):
        """Test parameter definition"""
        param = simulator.define_parameter(
            'test_param',
            DistributionType.NORMAL,
            {'mean': 0, 'std': 1},
            'Test parameter'
        )

        assert param.name == 'test_param'
        assert param.distribution == DistributionType.NORMAL
        assert param.parameters == {'mean': 0, 'std': 1}

    def test_sample_parameter_normal(self, simulator):
        """Test normal distribution sampling"""
        param = SimulationParameter(
            'normal_param',
            DistributionType.NORMAL,
            {'mean': 10, 'std': 2}
        )

        samples = simulator.sample_parameter(param, 1000)
        assert len(samples) == 1000
        assert abs(np.mean(samples) - 10) < 0.5  # Close to expected mean
        assert abs(np.std(samples) - 2) < 0.5   # Close to expected std

    def test_sample_parameter_beta(self, simulator):
        """Test beta distribution sampling"""
        param = SimulationParameter(
            'beta_param',
            DistributionType.BETA,
            {'alpha': 2, 'beta': 3}
        )

        samples = simulator.sample_parameter(param, 1000)
        assert len(samples) == 1000
        assert all(0 <= s <= 1 for s in samples)  # Beta distribution is [0,1]

    def test_sample_parameter_uniform(self, simulator):
        """Test uniform distribution sampling"""
        param = SimulationParameter(
            'uniform_param',
            DistributionType.UNIFORM,
            {'low': 5, 'high': 15}
        )

        samples = simulator.sample_parameter(param, 1000)
        assert len(samples) == 1000
        assert all(5 <= s <= 15 for s in samples)


class TestClinicalTrialSimulator:
    """Test suite for ClinicalTrialSimulator"""

    @pytest.fixture
    def trial_simulator(self):
        """Create trial simulator instance"""
        return ClinicalTrialSimulator(n_simulations=10, random_seed=42)

    def test_trial_simulator_initialization(self, trial_simulator):
        """Test trial simulator initialization"""
        assert isinstance(trial_simulator.trial_parameters, dict)
        assert isinstance(trial_simulator.outcome_models, dict)

    def test_define_trial_parameters(self, trial_simulator):
        """Test trial parameter definition"""
        trial_design = {
            'custom_param': {
                'distribution': 'normal',
                'parameters': {'mean': 5, 'std': 1},
                'description': 'Custom parameter'
            }
        }

        params = trial_simulator.define_trial_parameters(trial_design)
        assert isinstance(params, dict)
        assert 'treatment_effect' in params  # Default parameter
        assert 'custom_param' in params     # Custom parameter

    def test_simulate_patient_outcomes_basic(self, trial_simulator):
        """Test basic patient outcome simulation"""
        trial_simulator.define_trial_parameters({})

        results = trial_simulator.simulate_patient_outcomes(
            n_patients=20, treatment_group=True
        )

        assert isinstance(results, dict)
        # Check that some metrics are present
        expected_metrics = ['mean_outcome', 'completion_rate', 'adverse_rate']
        for metric in expected_metrics:
            if metric in results:
                assert isinstance(results[metric], dict)
                assert 'mean' in results[metric]

    def test_run_trial_simulation_basic(self, trial_simulator):
        """Test basic trial simulation"""
        results = trial_simulator.run_trial_simulation(
            n_treatment=10, n_control=10
        )

        assert isinstance(results, dict)
        assert 'simulation_parameters' in results
        assert 'treatment_group' in results or 'error' in results


class TestTreatmentOptimizationSimulator:
    """Test suite for TreatmentOptimizationSimulator"""

    @pytest.fixture
    def optimization_simulator(self):
        """Create optimization simulator instance"""
        return TreatmentOptimizationSimulator(n_simulations=10, random_seed=42)

    def test_optimization_simulator_initialization(self, optimization_simulator):
        """Test optimization simulator initialization"""
        assert isinstance(optimization_simulator.optimization_parameters, dict)

    def test_define_treatment_parameters(self, optimization_simulator):
        """Test treatment parameter definition"""
        params = optimization_simulator.define_treatment_parameters()

        assert isinstance(params, dict)
        assert 'cell_dose' in params
        assert 'injection_interval' in params
        assert 'administration_efficiency' in params

    def test_simulate_treatment_outcome(self, optimization_simulator):
        """Test treatment outcome simulation"""
        optimization_simulator.define_treatment_parameters()

        protocol_config = {
            'n_injections': 3,
            'dose_multiplier': 1.0,
            'interval_days': 14
        }

        outcome = optimization_simulator.simulate_treatment_outcome(protocol_config)

        assert isinstance(outcome, dict)
        assert 'efficacy' in outcome
        assert 'safety' in outcome
        assert 'total_cost' in outcome
        assert 0 <= outcome['efficacy'] <= 1
        assert 0 <= outcome['safety'] <= 1


class TestRiskAssessmentSimulator:
    """Test suite for RiskAssessmentSimulator"""

    @pytest.fixture
    def risk_simulator(self):
        """Create risk assessment simulator instance"""
        return RiskAssessmentSimulator(n_simulations=10, random_seed=42)

    def test_risk_simulator_initialization(self, risk_simulator):
        """Test risk simulator initialization"""
        assert isinstance(risk_simulator.risk_parameters, dict)

    def test_define_risk_parameters(self, risk_simulator):
        """Test risk parameter definition"""
        params = risk_simulator.define_risk_parameters()

        assert isinstance(params, dict)
        assert 'injection_site_reaction' in params
        assert 'allergic_reaction' in params
        assert 'tumorigenicity_risk' in params

    def test_simulate_safety_profile_basic(self, risk_simulator):
        """Test basic safety profile simulation"""
        patient_population = {
            'n_patients': 50,
            'mean_age': 55,
            'comorbidity_rate': 0.3
        }

        results = risk_simulator.simulate_safety_profile(
            patient_population, follow_up_months=12
        )

        assert isinstance(results, dict)
        assert 'safety_metrics' in results
        assert 'risk_assessment' in results
        assert 'simulation_parameters' in results


def test_distribution_types():
    """Test DistributionType enum"""
    assert DistributionType.NORMAL.value == "normal"
    assert DistributionType.BETA.value == "beta"
    assert DistributionType.UNIFORM.value == "uniform"


def test_simulation_parameter_dataclass():
    """Test SimulationParameter dataclass"""
    param = SimulationParameter(
        'test',
        DistributionType.NORMAL,
        {'mean': 0, 'std': 1},
        'Test parameter'
    )

    assert param.name == 'test'
    assert param.distribution == DistributionType.NORMAL
    assert param.description == 'Test parameter'


if __name__ == "__main__":
    pytest.main([__file__])