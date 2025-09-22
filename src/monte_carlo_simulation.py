"""
Monte Carlo Simulation Framework for Stem Cell Therapy Analysis

This module implements comprehensive Monte Carlo simulation models for:
1. Clinical trial outcome prediction with uncertainty quantification
2. Treatment protocol optimization under uncertainty
3. Risk assessment and safety analysis
4. Economic modeling and cost-effectiveness analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import json
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')


class DistributionType(Enum):
    """Supported probability distributions for Monte Carlo simulation"""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BINOMIAL = "binomial"
    POISSON = "poisson"


@dataclass
class SimulationParameter:
    """Parameter configuration for Monte Carlo simulation"""
    name: str
    distribution: DistributionType
    parameters: Dict[str, float]
    description: str = ""


class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulation framework for clinical trial analysis
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.simulation_results = {}
        self.parameter_history = {}
        self.convergence_analysis = {}

    def define_parameter(self, name: str, distribution: DistributionType,
                        parameters: Dict[str, float], description: str = "") -> SimulationParameter:
        """Define a simulation parameter with its probability distribution"""
        return SimulationParameter(
            name=name,
            distribution=distribution,
            parameters=parameters,
            description=description
        )

    def sample_parameter(self, param: SimulationParameter, n_samples: int = None) -> np.ndarray:
        """Sample values from a parameter's probability distribution"""
        if n_samples is None:
            n_samples = self.n_simulations

        if param.distribution == DistributionType.NORMAL:
            return np.random.normal(
                param.parameters['mean'],
                param.parameters['std'],
                n_samples
            )
        elif param.distribution == DistributionType.LOGNORMAL:
            return np.random.lognormal(
                param.parameters['mean'],
                param.parameters['std'],
                n_samples
            )
        elif param.distribution == DistributionType.BETA:
            return np.random.beta(
                param.parameters['alpha'],
                param.parameters['beta'],
                n_samples
            )
        elif param.distribution == DistributionType.GAMMA:
            return np.random.gamma(
                param.parameters['shape'],
                param.parameters['scale'],
                n_samples
            )
        elif param.distribution == DistributionType.UNIFORM:
            return np.random.uniform(
                param.parameters['low'],
                param.parameters['high'],
                n_samples
            )
        elif param.distribution == DistributionType.TRIANGULAR:
            return np.random.triangular(
                param.parameters['left'],
                param.parameters['mode'],
                param.parameters['right'],
                n_samples
            )
        elif param.distribution == DistributionType.BINOMIAL:
            return np.random.binomial(
                int(param.parameters['n']),
                param.parameters['p'],
                n_samples
            )
        elif param.distribution == DistributionType.POISSON:
            return np.random.poisson(
                param.parameters['lam'],
                n_samples
            )
        else:
            raise ValueError(f"Unsupported distribution: {param.distribution}")


class ClinicalTrialSimulator(MonteCarloSimulator):
    """
    Monte Carlo simulation for clinical trial outcomes
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        super().__init__(n_simulations, random_seed)
        self.trial_parameters = {}
        self.outcome_models = {}

    def define_trial_parameters(self, trial_design: Dict[str, Any]) -> Dict[str, SimulationParameter]:
        """Define clinical trial parameters for simulation"""

        # Standard trial parameters
        parameters = {
            # Efficacy parameters
            'treatment_effect': self.define_parameter(
                'treatment_effect',
                DistributionType.BETA,
                {'alpha': 5, 'beta': 3},
                'Primary treatment efficacy (0-1 scale)'
            ),

            # Safety parameters
            'adverse_event_rate': self.define_parameter(
                'adverse_event_rate',
                DistributionType.BETA,
                {'alpha': 2, 'beta': 20},
                'Rate of adverse events'
            ),

            # Population parameters
            'baseline_severity': self.define_parameter(
                'baseline_severity',
                DistributionType.NORMAL,
                {'mean': 50, 'std': 15},
                'Baseline disease severity score'
            ),

            # Follow-up parameters
            'dropout_rate': self.define_parameter(
                'dropout_rate',
                DistributionType.BETA,
                {'alpha': 2, 'beta': 8},
                'Patient dropout rate'
            ),

            # Time-dependent effects
            'time_to_effect': self.define_parameter(
                'time_to_effect',
                DistributionType.GAMMA,
                {'shape': 2, 'scale': 30},
                'Days until treatment effect observed'
            ),

            # Variability parameters
            'inter_patient_variability': self.define_parameter(
                'inter_patient_variability',
                DistributionType.LOGNORMAL,
                {'mean': 0, 'std': 0.5},
                'Between-patient response variability'
            )
        }

        # Update with custom parameters if provided
        if trial_design:
            for param_name, config in trial_design.items():
                if 'distribution' in config and 'parameters' in config:
                    parameters[param_name] = self.define_parameter(
                        param_name,
                        DistributionType(config['distribution']),
                        config['parameters'],
                        config.get('description', '')
                    )

        self.trial_parameters = parameters
        return parameters

    def simulate_patient_outcomes(self, n_patients: int,
                                 treatment_group: bool = True) -> Dict[str, np.ndarray]:
        """Simulate outcomes for a group of patients"""

        # Sample parameter values for this simulation
        treatment_effect = self.sample_parameter(
            self.trial_parameters['treatment_effect'],
            self.n_simulations
        )
        adverse_event_rate = self.sample_parameter(
            self.trial_parameters['adverse_event_rate'],
            self.n_simulations
        )
        baseline_severity = self.sample_parameter(
            self.trial_parameters['baseline_severity'],
            n_patients
        )
        dropout_rate = self.sample_parameter(
            self.trial_parameters['dropout_rate'],
            self.n_simulations
        )
        time_to_effect = self.sample_parameter(
            self.trial_parameters['time_to_effect'],
            self.n_simulations
        )
        variability = self.sample_parameter(
            self.trial_parameters['inter_patient_variability'],
            n_patients
        )

        results = {}

        for sim in range(self.n_simulations):
            # Simulate individual patient responses
            patient_outcomes = []

            for patient in range(n_patients):
                # Base outcome influenced by baseline severity
                base_outcome = baseline_severity[patient % len(baseline_severity)]

                if treatment_group:
                    # Apply treatment effect with variability
                    treatment_response = treatment_effect[sim] * variability[patient % len(variability)]
                    final_outcome = base_outcome + (100 - base_outcome) * treatment_response
                else:
                    # Control group - minimal improvement
                    placebo_effect = np.random.normal(0.05, 0.02)  # Small placebo effect
                    final_outcome = base_outcome + (100 - base_outcome) * placebo_effect

                # Apply dropout probability
                completes_study = np.random.random() > dropout_rate[sim]

                # Apply adverse events
                has_adverse_event = np.random.random() < adverse_event_rate[sim]

                patient_outcomes.append({
                    'outcome_score': final_outcome if completes_study else np.nan,
                    'completed_study': completes_study,
                    'adverse_event': has_adverse_event,
                    'time_to_effect': time_to_effect[sim] if treatment_group else np.inf,
                    'baseline_score': base_outcome
                })

            # Aggregate simulation results
            completed_patients = [p for p in patient_outcomes if p['completed_study']]

            if len(completed_patients) > 0:
                mean_outcome = np.mean([p['outcome_score'] for p in completed_patients])
                completion_rate = len(completed_patients) / n_patients
                adverse_rate = np.mean([p['adverse_event'] for p in patient_outcomes])
                mean_time_to_effect = np.mean([p['time_to_effect'] for p in completed_patients
                                             if np.isfinite(p['time_to_effect'])])
            else:
                mean_outcome = np.nan
                completion_rate = 0
                adverse_rate = 1
                mean_time_to_effect = np.inf

            # Store results for this simulation
            for key, value in {
                f'sim_{sim}_mean_outcome': mean_outcome,
                f'sim_{sim}_completion_rate': completion_rate,
                f'sim_{sim}_adverse_rate': adverse_rate,
                f'sim_{sim}_time_to_effect': mean_time_to_effect
            }.items():
                if key not in results:
                    results[key] = []
                results[key].append(value)

        # Convert to arrays and compute summary statistics
        summary_results = {}

        for metric in ['mean_outcome', 'completion_rate', 'adverse_rate', 'time_to_effect']:
            values = [results[f'sim_{sim}_{metric}'][0] for sim in range(self.n_simulations)
                     if f'sim_{sim}_{metric}' in results and len(results[f'sim_{sim}_{metric}']) > 0]
            values = [v for v in values if not np.isnan(v) and np.isfinite(v)]

            if len(values) > 0:
                summary_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': np.array(values)
                }

        return summary_results

    def run_trial_simulation(self, n_treatment: int, n_control: int,
                           trial_design: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete clinical trial simulation"""

        print(f"Running Monte Carlo simulation with {self.n_simulations} iterations...")

        # Define trial parameters
        self.define_trial_parameters(trial_design or {})

        # Simulate treatment and control groups
        print("Simulating treatment group...")
        treatment_results = self.simulate_patient_outcomes(n_treatment, treatment_group=True)

        print("Simulating control group...")
        control_results = self.simulate_patient_outcomes(n_control, treatment_group=False)

        # Calculate treatment effects
        treatment_effects = {}

        for metric in ['mean_outcome', 'completion_rate', 'adverse_rate']:
            if metric in treatment_results and metric in control_results:
                treatment_values = treatment_results[metric]['values']
                control_values = control_results[metric]['values']

                # Calculate differences
                if metric == 'adverse_rate':
                    # For adverse events, we want lower rates in treatment
                    differences = control_values - treatment_values
                else:
                    # For efficacy metrics, we want higher values in treatment
                    differences = treatment_values - control_values

                # Effect size calculation
                pooled_std = np.sqrt((np.var(treatment_values) + np.var(control_values)) / 2)
                effect_size = np.mean(differences) / pooled_std if pooled_std > 0 else 0

                # Statistical significance simulation
                p_values = []
                for i in range(len(treatment_values)):
                    t_stat = differences[i] / (pooled_std * np.sqrt(2/min(n_treatment, n_control)))
                    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                    p_values.append(p_val)

                treatment_effects[f'{metric}_effect'] = {
                    'mean_difference': np.mean(differences),
                    'std_difference': np.std(differences),
                    'effect_size': effect_size,
                    'probability_positive': np.mean(differences > 0),
                    'probability_significant': np.mean(np.array(p_values) < 0.05),
                    'confidence_interval_95': [
                        np.percentile(differences, 2.5),
                        np.percentile(differences, 97.5)
                    ]
                }

        # Trial success probability
        success_criteria = {
            'efficacy_significant': treatment_effects.get('mean_outcome_effect', {}).get('probability_significant', 0),
            'safety_acceptable': treatment_effects.get('adverse_rate_effect', {}).get('probability_positive', 0),
            'completion_adequate': np.mean(treatment_results['completion_rate']['values'] > 0.8)
        }

        overall_success = (
            success_criteria['efficacy_significant'] > 0.8 and
            success_criteria['safety_acceptable'] > 0.7 and
            success_criteria['completion_adequate'] > 0.6
        )

        results = {
            'treatment_group': treatment_results,
            'control_group': control_results,
            'treatment_effects': treatment_effects,
            'success_criteria': success_criteria,
            'overall_success_probability': overall_success,
            'simulation_parameters': {
                'n_simulations': self.n_simulations,
                'n_treatment': n_treatment,
                'n_control': n_control,
                'random_seed': self.random_seed
            }
        }

        self.simulation_results['clinical_trial'] = results
        return results


class TreatmentOptimizationSimulator(MonteCarloSimulator):
    """
    Monte Carlo simulation for treatment protocol optimization
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        super().__init__(n_simulations, random_seed)
        self.optimization_parameters = {}

    def define_treatment_parameters(self) -> Dict[str, SimulationParameter]:
        """Define treatment protocol parameters for optimization"""

        parameters = {
            # Dose parameters
            'cell_dose': self.define_parameter(
                'cell_dose',
                DistributionType.LOGNORMAL,
                {'mean': np.log(1e6), 'std': 0.5},
                'Cell dose (cells per injection)'
            ),

            # Timing parameters
            'injection_interval': self.define_parameter(
                'injection_interval',
                DistributionType.GAMMA,
                {'shape': 2, 'scale': 7},
                'Days between injections'
            ),

            # Route of administration effect
            'administration_efficiency': self.define_parameter(
                'administration_efficiency',
                DistributionType.BETA,
                {'alpha': 8, 'beta': 2},
                'Efficiency of administration route'
            ),

            # Patient characteristics
            'patient_age': self.define_parameter(
                'patient_age',
                DistributionType.NORMAL,
                {'mean': 55, 'std': 12},
                'Patient age in years'
            ),

            'disease_severity': self.define_parameter(
                'disease_severity',
                DistributionType.BETA,
                {'alpha': 3, 'beta': 3},
                'Disease severity (0-1 scale)'
            ),

            # Cost parameters
            'treatment_cost_per_dose': self.define_parameter(
                'treatment_cost_per_dose',
                DistributionType.LOGNORMAL,
                {'mean': np.log(50000), 'std': 0.3},
                'Cost per treatment dose (USD)'
            )
        }

        self.optimization_parameters = parameters
        return parameters

    def simulate_treatment_outcome(self, protocol_config: Dict[str, float]) -> Dict[str, float]:
        """Simulate outcome for a specific treatment protocol"""

        # Sample parameter values
        params = {}
        for param_name, param_obj in self.optimization_parameters.items():
            params[param_name] = self.sample_parameter(param_obj, 1)[0]

        # Extract protocol configuration
        n_injections = protocol_config.get('n_injections', 3)
        dose_multiplier = protocol_config.get('dose_multiplier', 1.0)
        interval_days = protocol_config.get('interval_days', 14)

        # Calculate total dose
        total_dose = params['cell_dose'] * dose_multiplier * n_injections

        # Dose-response relationship (sigmoid curve)
        optimal_dose = 3e6  # Optimal dose
        dose_effect = total_dose / (total_dose + optimal_dose)

        # Administration efficiency effect
        admin_effect = params['administration_efficiency']

        # Timing effect (optimal interval around 14 days)
        timing_effect = np.exp(-0.01 * (interval_days - 14)**2)

        # Patient factors
        age_effect = np.exp(-0.01 * max(0, params['patient_age'] - 50))
        severity_effect = 1 - 0.3 * params['disease_severity']  # More severe = harder to treat

        # Calculate efficacy (0-1 scale)
        efficacy = dose_effect * admin_effect * timing_effect * age_effect * severity_effect
        efficacy = np.clip(efficacy, 0, 1)

        # Calculate safety (inverse relationship with dose)
        safety_risk = np.log(total_dose / 1e6) * 0.05  # 5% risk per log unit above 1M cells
        safety = 1 - np.clip(safety_risk, 0, 0.3)  # Max 30% safety issues

        # Calculate cost
        total_cost = params['treatment_cost_per_dose'] * n_injections

        # Quality-adjusted outcome
        quality_adjusted_efficacy = efficacy * safety

        # Cost-effectiveness ratio
        cost_effectiveness = total_cost / (quality_adjusted_efficacy + 0.01)  # Avoid division by zero

        return {
            'efficacy': efficacy,
            'safety': safety,
            'total_cost': total_cost,
            'quality_adjusted_efficacy': quality_adjusted_efficacy,
            'cost_effectiveness': cost_effectiveness,
            'total_dose': total_dose
        }

    def optimize_treatment_protocol(self, objective: str = 'quality_adjusted_efficacy',
                                  constraints: Dict[str, float] = None) -> Dict[str, Any]:
        """Optimize treatment protocol using Monte Carlo simulation"""

        print(f"Optimizing treatment protocol for: {objective}")

        # Define parameters
        self.define_treatment_parameters()

        # Define protocol search space
        protocol_ranges = {
            'n_injections': (1, 5),
            'dose_multiplier': (0.5, 3.0),
            'interval_days': (7, 28)
        }

        constraints = constraints or {}

        best_protocols = []

        # Grid search with Monte Carlo evaluation
        n_injections_range = range(int(protocol_ranges['n_injections'][0]),
                                 int(protocol_ranges['n_injections'][1]) + 1)
        dose_multipliers = np.linspace(protocol_ranges['dose_multiplier'][0],
                                     protocol_ranges['dose_multiplier'][1], 10)
        intervals = np.linspace(protocol_ranges['interval_days'][0],
                              protocol_ranges['interval_days'][1], 10)

        for n_inj in n_injections_range:
            for dose_mult in dose_multipliers:
                for interval in intervals:
                    protocol = {
                        'n_injections': n_inj,
                        'dose_multiplier': dose_mult,
                        'interval_days': interval
                    }

                    # Run Monte Carlo simulation for this protocol
                    outcomes = []
                    for _ in range(100):  # Smaller sample for optimization
                        outcome = self.simulate_treatment_outcome(protocol)
                        outcomes.append(outcome)

                    # Calculate average outcomes
                    avg_outcomes = {}
                    for key in outcomes[0].keys():
                        values = [o[key] for o in outcomes]
                        avg_outcomes[key] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'median': np.median(values)
                        }

                    # Check constraints
                    meets_constraints = True
                    for constraint_key, constraint_value in constraints.items():
                        if constraint_key in avg_outcomes:
                            if avg_outcomes[constraint_key]['mean'] < constraint_value:
                                meets_constraints = False
                                break

                    if meets_constraints:
                        best_protocols.append({
                            'protocol': protocol,
                            'outcomes': avg_outcomes,
                            'objective_value': avg_outcomes[objective]['mean']
                        })

        # Sort by objective
        if objective == 'cost_effectiveness':
            # Lower is better for cost-effectiveness
            best_protocols.sort(key=lambda x: x['objective_value'])
        else:
            # Higher is better for efficacy metrics
            best_protocols.sort(key=lambda x: x['objective_value'], reverse=True)

        # Detailed analysis of top protocols
        top_protocols = best_protocols[:5]

        for i, protocol_result in enumerate(top_protocols):
            print(f"\nTop Protocol #{i+1}:")
            print(f"  Configuration: {protocol_result['protocol']}")
            print(f"  {objective}: {protocol_result['objective_value']:.4f}")
            if 'efficacy' in protocol_result['outcomes']:
                print(f"  Efficacy: {protocol_result['outcomes']['efficacy']['mean']:.4f}")
            if 'safety' in protocol_result['outcomes']:
                print(f"  Safety: {protocol_result['outcomes']['safety']['mean']:.4f}")
            if 'total_cost' in protocol_result['outcomes']:
                print(f"  Cost: ${protocol_result['outcomes']['total_cost']['mean']:,.0f}")

        results = {
            'best_protocols': top_protocols,
            'all_evaluated_protocols': best_protocols,
            'optimization_objective': objective,
            'constraints': constraints,
            'parameter_ranges': protocol_ranges
        }

        self.simulation_results['treatment_optimization'] = results
        return results


class RiskAssessmentSimulator(MonteCarloSimulator):
    """
    Monte Carlo simulation for risk assessment and safety analysis
    """

    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        super().__init__(n_simulations, random_seed)
        self.risk_parameters = {}

    def define_risk_parameters(self) -> Dict[str, SimulationParameter]:
        """Define risk parameters for safety analysis"""

        parameters = {
            # Immediate risks
            'injection_site_reaction': self.define_parameter(
                'injection_site_reaction',
                DistributionType.BETA,
                {'alpha': 5, 'beta': 20},
                'Probability of injection site reaction'
            ),

            'allergic_reaction': self.define_parameter(
                'allergic_reaction',
                DistributionType.BETA,
                {'alpha': 1, 'beta': 200},
                'Probability of allergic reaction'
            ),

            # Short-term risks
            'infection_risk': self.define_parameter(
                'infection_risk',
                DistributionType.BETA,
                {'alpha': 2, 'beta': 100},
                'Probability of infection'
            ),

            'immune_response': self.define_parameter(
                'immune_response',
                DistributionType.BETA,
                {'alpha': 8, 'beta': 12},
                'Probability of immune response'
            ),

            # Long-term risks
            'tumorigenicity_risk': self.define_parameter(
                'tumorigenicity_risk',
                DistributionType.BETA,
                {'alpha': 1, 'beta': 1000},
                'Long-term tumorigenicity risk'
            ),

            'immune_system_effects': self.define_parameter(
                'immune_system_effects',
                DistributionType.BETA,
                {'alpha': 3, 'beta': 30},
                'Long-term immune system effects'
            ),

            # Risk modifiers
            'patient_age_risk_modifier': self.define_parameter(
                'patient_age_risk_modifier',
                DistributionType.LOGNORMAL,
                {'mean': 0, 'std': 0.2},
                'Age-related risk modification factor'
            ),

            'comorbidity_risk_modifier': self.define_parameter(
                'comorbidity_risk_modifier',
                DistributionType.LOGNORMAL,
                {'mean': 0.2, 'std': 0.3},
                'Comorbidity-related risk modification'
            )
        }

        self.risk_parameters = parameters
        return parameters

    def simulate_safety_profile(self, patient_population: Dict[str, Any],
                               follow_up_months: int = 24) -> Dict[str, Any]:
        """Simulate safety profile for a patient population"""

        print(f"Simulating safety profile for {follow_up_months} months follow-up...")

        # Define risk parameters
        self.define_risk_parameters()

        n_patients = patient_population.get('n_patients', 100)
        mean_age = patient_population.get('mean_age', 55)
        comorbidity_rate = patient_population.get('comorbidity_rate', 0.3)

        safety_results = {}

        for sim in range(self.n_simulations):
            # Sample risk parameters for this simulation
            risks = {}
            for param_name, param_obj in self.risk_parameters.items():
                risks[param_name] = self.sample_parameter(param_obj, 1)[0]

            # Simulate individual patient outcomes
            patient_events = []

            for patient in range(n_patients):
                # Patient characteristics
                patient_age = np.random.normal(mean_age, 10)
                has_comorbidity = np.random.random() < comorbidity_rate

                # Age risk modifier (higher risk for older patients)
                age_modifier = risks['patient_age_risk_modifier'] * max(0, (patient_age - 40) / 40)

                # Comorbidity risk modifier
                comorbidity_modifier = risks['comorbidity_risk_modifier'] if has_comorbidity else 1.0

                # Calculate individual risk probabilities
                individual_risks = {}
                for risk_name in ['injection_site_reaction', 'allergic_reaction',
                                'infection_risk', 'immune_response']:
                    base_risk = risks[risk_name]
                    modified_risk = base_risk * (1 + age_modifier) * comorbidity_modifier
                    individual_risks[risk_name] = min(modified_risk, 0.95)  # Cap at 95%

                # Long-term risks (cumulative over follow-up period)
                for risk_name in ['tumorigenicity_risk', 'immune_system_effects']:
                    base_risk = risks[risk_name]
                    # Risk accumulates over time
                    cumulative_risk = 1 - (1 - base_risk) ** (follow_up_months / 12)
                    modified_risk = cumulative_risk * (1 + age_modifier) * comorbidity_modifier
                    individual_risks[risk_name] = min(modified_risk, 0.8)  # Cap at 80%

                # Simulate events for this patient
                patient_events_dict = {}
                for risk_name, risk_prob in individual_risks.items():
                    patient_events_dict[risk_name] = np.random.random() < risk_prob

                patient_events_dict['age'] = patient_age
                patient_events_dict['has_comorbidity'] = has_comorbidity
                patient_events.append(patient_events_dict)

            # Aggregate results for this simulation
            sim_results = {}

            # Calculate event rates
            for risk_name in ['injection_site_reaction', 'allergic_reaction',
                            'infection_risk', 'immune_response',
                            'tumorigenicity_risk', 'immune_system_effects']:
                events = [p[risk_name] for p in patient_events]
                sim_results[f'{risk_name}_rate'] = np.mean(events)

            # Calculate serious adverse event rate
            serious_events = []
            for p in patient_events:
                serious = (p['allergic_reaction'] or p['infection_risk'] or
                          p['tumorigenicity_risk'])
                serious_events.append(serious)
            sim_results['serious_adverse_event_rate'] = np.mean(serious_events)

            # Risk by subgroups
            elderly_patients = [p for p in patient_events if p['age'] > 65]
            if elderly_patients:
                sim_results['elderly_serious_ae_rate'] = np.mean([
                    p['allergic_reaction'] or p['infection_risk'] or p['tumorigenicity_risk']
                    for p in elderly_patients
                ])

            comorbid_patients = [p for p in patient_events if p['has_comorbidity']]
            if comorbid_patients:
                sim_results['comorbid_serious_ae_rate'] = np.mean([
                    p['allergic_reaction'] or p['infection_risk'] or p['tumorigenicity_risk']
                    for p in comorbid_patients
                ])

            # Store simulation results
            for key, value in sim_results.items():
                if key not in safety_results:
                    safety_results[key] = []
                safety_results[key].append(value)

        # Calculate summary statistics
        summary_results = {}
        for metric, values in safety_results.items():
            summary_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'q95': np.percentile(values, 95),
                'q99': np.percentile(values, 99),
                'values': np.array(values)
            }

        # Risk categorization
        overall_risk_assessment = self._categorize_risk(summary_results)

        results = {
            'safety_metrics': summary_results,
            'risk_assessment': overall_risk_assessment,
            'simulation_parameters': {
                'n_simulations': self.n_simulations,
                'n_patients': n_patients,
                'follow_up_months': follow_up_months,
                'population_characteristics': patient_population
            }
        }

        self.simulation_results['risk_assessment'] = results
        return results

    def _categorize_risk(self, summary_results: Dict[str, Dict]) -> Dict[str, str]:
        """Categorize overall risk level based on simulation results"""

        risk_assessment = {}

        # Serious adverse event rate assessment
        serious_ae_rate = summary_results.get('serious_adverse_event_rate', {}).get('q95', 0)
        if serious_ae_rate < 0.05:
            risk_assessment['serious_ae_risk'] = 'Low'
        elif serious_ae_rate < 0.15:
            risk_assessment['serious_ae_risk'] = 'Moderate'
        else:
            risk_assessment['serious_ae_risk'] = 'High'

        # Tumorigenicity risk assessment
        tumor_risk = summary_results.get('tumorigenicity_risk_rate', {}).get('q95', 0)
        if tumor_risk < 0.01:
            risk_assessment['tumorigenicity_risk'] = 'Low'
        elif tumor_risk < 0.05:
            risk_assessment['tumorigenicity_risk'] = 'Moderate'
        else:
            risk_assessment['tumorigenicity_risk'] = 'High'

        # Overall risk level
        if all(level in ['Low'] for level in risk_assessment.values()):
            risk_assessment['overall_risk'] = 'Low'
        elif any(level == 'High' for level in risk_assessment.values()):
            risk_assessment['overall_risk'] = 'High'
        else:
            risk_assessment['overall_risk'] = 'Moderate'

        return risk_assessment


def main():
    """Main function to demonstrate Monte Carlo simulation capabilities"""

    print("="*60)
    print("MONTE CARLO SIMULATION FRAMEWORK FOR STEM CELL THERAPY")
    print("="*60)

    # Clinical Trial Simulation
    print("\n1. CLINICAL TRIAL SIMULATION")
    print("-" * 40)

    trial_simulator = ClinicalTrialSimulator(n_simulations=1000, random_seed=42)

    # Custom trial design for heart failure study
    trial_design = {
        'treatment_effect': {
            'distribution': 'beta',
            'parameters': {'alpha': 6, 'beta': 2},  # More optimistic
            'description': 'Heart failure treatment efficacy'
        },
        'adverse_event_rate': {
            'distribution': 'beta',
            'parameters': {'alpha': 3, 'beta': 25},  # Lower AE rate
            'description': 'Cardiovascular adverse events'
        }
    }

    trial_results = trial_simulator.run_trial_simulation(
        n_treatment=150,
        n_control=150,
        trial_design=trial_design
    )

    print(f"Efficacy success probability: {trial_results['success_criteria']['efficacy_significant']:.3f}")
    print(f"Safety success probability: {trial_results['success_criteria']['safety_acceptable']:.3f}")
    print(f"Overall trial success probability: {trial_results['overall_success_probability']}")

    # Treatment Optimization Simulation
    print("\n2. TREATMENT OPTIMIZATION SIMULATION")
    print("-" * 40)

    optimization_simulator = TreatmentOptimizationSimulator(n_simulations=1000, random_seed=42)

    # Optimize for quality-adjusted efficacy with cost constraint
    constraints = {
        'safety': 0.8,  # Minimum 80% safety
        'total_cost': 200000  # Maximum $200k cost
    }

    optimization_results = optimization_simulator.optimize_treatment_protocol(
        objective='quality_adjusted_efficacy',
        constraints=constraints
    )

    # Risk Assessment Simulation
    print("\n3. RISK ASSESSMENT SIMULATION")
    print("-" * 40)

    risk_simulator = RiskAssessmentSimulator(n_simulations=1000, random_seed=42)

    # Simulate elderly population with comorbidities
    elderly_population = {
        'n_patients': 200,
        'mean_age': 68,
        'comorbidity_rate': 0.6
    }

    risk_results = risk_simulator.simulate_safety_profile(
        patient_population=elderly_population,
        follow_up_months=36
    )

    print(f"Serious AE rate (95% CI): {risk_results['safety_metrics']['serious_adverse_event_rate']['q95']:.4f}")
    print(f"Overall risk assessment: {risk_results['risk_assessment']['overall_risk']}")
    print(f"Tumorigenicity risk: {risk_results['risk_assessment']['tumorigenicity_risk']}")

    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION ANALYSIS COMPLETED!")
    print("="*60)

    return {
        'clinical_trial': trial_results,
        'treatment_optimization': optimization_results,
        'risk_assessment': risk_results
    }


if __name__ == "__main__":
    results = main()