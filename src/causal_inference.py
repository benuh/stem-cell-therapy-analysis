"""
Causal Inference Framework for Stem Cell Therapy Analysis

This module implements advanced causal inference methods to understand:
1. Causal relationships between treatment variables and outcomes
2. Confounding effects and bias adjustment
3. Treatment effect heterogeneity
4. Mediation analysis and causal pathways
5. Instrumental variable analysis
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
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
warnings.filterwarnings('ignore')


class CausalIdentificationStrategy(Enum):
    """Causal identification strategies"""
    RANDOMIZED_EXPERIMENT = "randomized_experiment"
    PROPENSITY_SCORE = "propensity_score"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    MATCHING = "matching"
    DOUBLY_ROBUST = "doubly_robust"


@dataclass
class CausalEstimate:
    """Structure for causal effect estimates"""
    treatment: str
    outcome: str
    effect_size: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    sample_size: int
    confounders_adjusted: List[str]


class CausalDAG:
    """
    Directed Acyclic Graph for causal relationships
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.variables = {}
        self.edges = []

    def add_variable(self, name: str, variable_type: str = "observed",
                    description: str = "") -> None:
        """Add a variable to the causal graph"""
        self.graph.add_node(name)
        self.variables[name] = {
            'type': variable_type,  # observed, latent, treatment, outcome
            'description': description
        }

    def add_causal_edge(self, cause: str, effect: str, edge_type: str = "direct") -> None:
        """Add a causal edge between variables"""
        if cause not in self.graph.nodes:
            self.add_variable(cause)
        if effect not in self.graph.nodes:
            self.add_variable(effect)

        self.graph.add_edge(cause, effect)
        self.edges.append({
            'cause': cause,
            'effect': effect,
            'type': edge_type
        })

    def get_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Identify confounding variables using backdoor criterion"""
        # Find all paths from treatment to outcome
        try:
            all_paths = list(nx.all_simple_paths(self.graph.to_undirected(),
                                               treatment, outcome))
        except nx.NetworkXNoPath:
            return []

        # Identify backdoor paths (paths that start with an edge into treatment)
        backdoor_paths = []
        for path in all_paths:
            if len(path) > 2:  # Must have intermediate nodes
                # Check if path starts with edge into treatment
                if self.graph.has_edge(path[1], path[0]):
                    backdoor_paths.append(path)

        # Extract confounders (nodes that can block backdoor paths)
        confounders = set()
        for path in backdoor_paths:
            # Add intermediate nodes as potential confounders
            confounders.update(path[1:-1])

        return list(confounders)

    def get_mediators(self, treatment: str, outcome: str) -> List[str]:
        """Identify mediating variables"""
        try:
            direct_paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
        except nx.NetworkXNoPath:
            return []

        mediators = set()
        for path in direct_paths:
            if len(path) > 2:  # Has intermediate nodes
                mediators.update(path[1:-1])

        return list(mediators)

    def visualize_dag(self) -> go.Figure:
        """Create interactive visualization of the causal DAG"""
        pos = nx.spring_layout(self.graph, seed=42)

        # Extract node and edge information
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        node_text = list(self.graph.nodes())

        # Color nodes by type
        node_colors = []
        for node in self.graph.nodes():
            var_type = self.variables.get(node, {}).get('type', 'observed')
            if var_type == 'treatment':
                node_colors.append('lightblue')
            elif var_type == 'outcome':
                node_colors.append('lightcoral')
            elif var_type == 'latent':
                node_colors.append('lightgray')
            else:
                node_colors.append('lightgreen')

        # Create edges
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=2, color='gray'),
                               hoverinfo='none',
                               mode='lines'))

        # Add nodes
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               marker=dict(size=30,
                                         color=node_colors,
                                         line=dict(width=2, color='black'))))

        fig.update_layout(title="Causal Directed Acyclic Graph (DAG)",
                         showlegend=False,
                         hovermode='closest',
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        return fig


class PropensityScoreAnalysis:
    """
    Propensity score methods for causal inference
    """

    def __init__(self):
        self.propensity_model = None
        self.propensity_scores = None
        self.matched_data = None

    def estimate_propensity_scores(self, data: pd.DataFrame, treatment_col: str,
                                 confounder_cols: List[str]) -> np.ndarray:
        """Estimate propensity scores using logistic regression"""

        X = data[confounder_cols].copy()
        y = data[treatment_col].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression
        self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        self.propensity_model.fit(X_scaled, y)

        # Calculate propensity scores
        self.propensity_scores = self.propensity_model.predict_proba(X_scaled)[:, 1]

        return self.propensity_scores

    def propensity_score_matching(self, data: pd.DataFrame, treatment_col: str,
                                outcome_col: str, confounder_cols: List[str],
                                caliper: float = 0.1) -> pd.DataFrame:
        """Perform 1:1 propensity score matching"""

        # Estimate propensity scores
        ps = self.estimate_propensity_scores(data, treatment_col, confounder_cols)
        data_with_ps = data.copy()
        data_with_ps['propensity_score'] = ps

        # Separate treated and control groups
        treated = data_with_ps[data_with_ps[treatment_col] == 1].copy()
        control = data_with_ps[data_with_ps[treatment_col] == 0].copy()

        matched_pairs = []

        # For each treated unit, find closest control unit within caliper
        for _, treated_unit in treated.iterrows():
            treated_ps = treated_unit['propensity_score']

            # Calculate distances to all control units
            control['ps_distance'] = np.abs(control['propensity_score'] - treated_ps)

            # Find closest match within caliper
            potential_matches = control[control['ps_distance'] <= caliper]

            if len(potential_matches) > 0:
                # Select closest match
                best_match = potential_matches.loc[potential_matches['ps_distance'].idxmin()]

                # Add matched pair
                matched_pairs.append(treated_unit.to_dict())
                matched_pairs.append(best_match.to_dict())

                # Remove matched control unit
                control = control.drop(best_match.name)

        self.matched_data = pd.DataFrame(matched_pairs)
        return self.matched_data

    def estimate_treatment_effect_matching(self, outcome_col: str,
                                         treatment_col: str) -> CausalEstimate:
        """Estimate treatment effect using matched data"""

        if self.matched_data is None:
            raise ValueError("Must perform matching first")

        treated = self.matched_data[self.matched_data[treatment_col] == 1]
        control = self.matched_data[self.matched_data[treatment_col] == 0]

        # Calculate average treatment effect
        ate = treated[outcome_col].mean() - control[outcome_col].mean()

        # Calculate standard error (simplified)
        treated_var = treated[outcome_col].var()
        control_var = control[outcome_col].var()
        se = np.sqrt(treated_var / len(treated) + control_var / len(control))

        # Confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # P-value
        t_stat = ate / se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return CausalEstimate(
            treatment=treatment_col,
            outcome=outcome_col,
            effect_size=ate,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="propensity_score_matching",
            sample_size=len(self.matched_data),
            confounders_adjusted=[]
        )

    def ipw_estimation(self, data: pd.DataFrame, treatment_col: str,
                      outcome_col: str, confounder_cols: List[str]) -> CausalEstimate:
        """Inverse Probability Weighting estimation"""

        # Estimate propensity scores
        ps = self.estimate_propensity_scores(data, treatment_col, confounder_cols)

        # Calculate IPW weights
        weights = np.where(data[treatment_col] == 1, 1/ps, 1/(1-ps))

        # Weighted average outcomes
        treated_outcome = np.average(
            data[data[treatment_col] == 1][outcome_col],
            weights=weights[data[treatment_col] == 1]
        )

        control_outcome = np.average(
            data[data[treatment_col] == 0][outcome_col],
            weights=weights[data[treatment_col] == 0]
        )

        # Treatment effect
        ate = treated_outcome - control_outcome

        # Standard error calculation (simplified)
        n_treated = sum(data[treatment_col] == 1)
        n_control = sum(data[treatment_col] == 0)
        se = np.sqrt(
            data[data[treatment_col] == 1][outcome_col].var() / n_treated +
            data[data[treatment_col] == 0][outcome_col].var() / n_control
        )

        # Confidence interval and p-value
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        p_value = 2 * (1 - stats.norm.cdf(abs(ate / se)))

        return CausalEstimate(
            treatment=treatment_col,
            outcome=outcome_col,
            effect_size=ate,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="inverse_probability_weighting",
            sample_size=len(data),
            confounders_adjusted=confounder_cols
        )


class InstrumentalVariableAnalysis:
    """
    Instrumental Variable methods for causal inference
    """

    def __init__(self):
        self.first_stage_model = None
        self.second_stage_model = None

    def two_stage_least_squares(self, data: pd.DataFrame, treatment_col: str,
                               outcome_col: str, instrument_col: str,
                               control_cols: List[str] = None) -> CausalEstimate:
        """Two-Stage Least Squares estimation"""

        if control_cols is None:
            control_cols = []

        # Prepare data
        X_controls = data[control_cols] if control_cols else pd.DataFrame(index=data.index)
        X_controls = sm.add_constant(X_controls)  # Add intercept

        # First stage: Instrument -> Treatment
        X_first_stage = data[[instrument_col] + control_cols]
        X_first_stage = sm.add_constant(X_first_stage)

        self.first_stage_model = sm.OLS(data[treatment_col], X_first_stage).fit()
        treatment_predicted = self.first_stage_model.fittedvalues

        # Check instrument strength (F-statistic > 10 rule of thumb)
        f_stat = self.first_stage_model.fvalue
        if f_stat < 10:
            warnings.warn(f"Weak instrument warning: F-statistic = {f_stat:.2f} < 10")

        # Second stage: Predicted Treatment -> Outcome
        X_second_stage = pd.concat([
            pd.Series(treatment_predicted, name=treatment_col),
            X_controls.drop('const', axis=1)
        ], axis=1)
        X_second_stage = sm.add_constant(X_second_stage)

        self.second_stage_model = sm.OLS(data[outcome_col], X_second_stage).fit()

        # Extract treatment effect
        treatment_effect = self.second_stage_model.params[treatment_col]
        se = self.second_stage_model.bse[treatment_col]

        # Confidence interval
        ci_lower = treatment_effect - 1.96 * se
        ci_upper = treatment_effect + 1.96 * se

        # P-value
        p_value = self.second_stage_model.pvalues[treatment_col]

        return CausalEstimate(
            treatment=treatment_col,
            outcome=outcome_col,
            effect_size=treatment_effect,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="two_stage_least_squares",
            sample_size=len(data),
            confounders_adjusted=control_cols
        )

    def validate_instrument(self, data: pd.DataFrame, treatment_col: str,
                          outcome_col: str, instrument_col: str) -> Dict[str, Any]:
        """Validate instrumental variable assumptions"""

        validation_results = {}

        # 1. Relevance: Instrument should predict treatment
        corr_instrument_treatment = data[instrument_col].corr(data[treatment_col])
        validation_results['relevance_correlation'] = corr_instrument_treatment

        # F-test for instrument strength
        X = sm.add_constant(data[instrument_col])
        model = sm.OLS(data[treatment_col], X).fit()
        f_stat = model.fvalue
        validation_results['first_stage_f_statistic'] = f_stat
        validation_results['instrument_strength'] = 'Strong' if f_stat > 10 else 'Weak'

        # 2. Exclusion restriction test (indirect - correlation with outcome)
        corr_instrument_outcome = data[instrument_col].corr(data[outcome_col])
        validation_results['exclusion_violation_indicator'] = corr_instrument_outcome

        # 3. Independence assumption (cannot be directly tested)
        validation_results['independence_assumption'] = 'Cannot be directly tested'

        return validation_results


class MediationAnalysis:
    """
    Mediation analysis to understand causal pathways
    """

    def __init__(self):
        self.mediation_results = {}

    def causal_mediation_analysis(self, data: pd.DataFrame, treatment_col: str,
                                mediator_col: str, outcome_col: str,
                                control_cols: List[str] = None) -> Dict[str, Any]:
        """Perform causal mediation analysis"""

        if control_cols is None:
            control_cols = []

        # Prepare control variables
        controls = data[control_cols] if control_cols else pd.DataFrame(index=data.index)

        # Model 1: Treatment -> Mediator
        X_med = pd.concat([data[[treatment_col]], controls], axis=1)
        X_med = sm.add_constant(X_med)
        mediator_model = sm.OLS(data[mediator_col], X_med).fit()

        # Model 2: Treatment + Mediator -> Outcome
        X_out = pd.concat([data[[treatment_col, mediator_col]], controls], axis=1)
        X_out = sm.add_constant(X_out)
        outcome_model = sm.OLS(data[outcome_col], X_out).fit()

        # Model 3: Treatment -> Outcome (total effect)
        X_total = pd.concat([data[[treatment_col]], controls], axis=1)
        X_total = sm.add_constant(X_total)
        total_model = sm.OLS(data[outcome_col], X_total).fit()

        # Extract coefficients
        alpha = mediator_model.params[treatment_col]  # Treatment -> Mediator
        beta = outcome_model.params[mediator_col]     # Mediator -> Outcome
        tau_prime = outcome_model.params[treatment_col]  # Direct effect
        tau = total_model.params[treatment_col]       # Total effect

        # Calculate mediation effects
        indirect_effect = alpha * beta  # Mediated effect
        direct_effect = tau_prime       # Direct effect
        total_effect = tau             # Total effect

        # Proportion mediated
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0

        # Bootstrap confidence intervals for indirect effect
        indirect_effects = []
        n_bootstrap = 1000

        for _ in range(n_bootstrap):
            # Resample data
            boot_data = data.sample(n=len(data), replace=True)

            # Fit models on bootstrap sample
            X_med_boot = pd.concat([boot_data[[treatment_col]],
                                  boot_data[control_cols] if control_cols else pd.DataFrame(index=boot_data.index)], axis=1)
            X_med_boot = sm.add_constant(X_med_boot)
            med_model_boot = sm.OLS(boot_data[mediator_col], X_med_boot).fit()

            X_out_boot = pd.concat([boot_data[[treatment_col, mediator_col]],
                                  boot_data[control_cols] if control_cols else pd.DataFrame(index=boot_data.index)], axis=1)
            X_out_boot = sm.add_constant(X_out_boot)
            out_model_boot = sm.OLS(boot_data[outcome_col], X_out_boot).fit()

            # Calculate indirect effect
            alpha_boot = med_model_boot.params[treatment_col]
            beta_boot = out_model_boot.params[mediator_col]
            indirect_effects.append(alpha_boot * beta_boot)

        # Bootstrap confidence interval
        ci_lower = np.percentile(indirect_effects, 2.5)
        ci_upper = np.percentile(indirect_effects, 97.5)

        results = {
            'total_effect': total_effect,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'proportion_mediated': proportion_mediated,
            'indirect_effect_ci': (ci_lower, ci_upper),
            'models': {
                'mediator_model': mediator_model,
                'outcome_model': outcome_model,
                'total_effect_model': total_model
            },
            'coefficients': {
                'alpha': alpha,  # Treatment -> Mediator
                'beta': beta,    # Mediator -> Outcome
                'tau_prime': tau_prime,  # Direct effect
                'tau': tau       # Total effect
            }
        }

        self.mediation_results = results
        return results


class DoublyRobustEstimation:
    """
    Doubly robust estimation combining propensity scores and outcome regression
    """

    def __init__(self):
        self.propensity_model = None
        self.outcome_model = None

    def doubly_robust_ate(self, data: pd.DataFrame, treatment_col: str,
                         outcome_col: str, confounder_cols: List[str]) -> CausalEstimate:
        """Doubly robust average treatment effect estimation"""

        # Step 1: Estimate propensity scores
        X = data[confounder_cols].fillna(data[confounder_cols].median())
        X_scaled = StandardScaler().fit_transform(X)

        self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        self.propensity_model.fit(X_scaled, data[treatment_col])
        ps = self.propensity_model.predict_proba(X_scaled)[:, 1]

        # Step 2: Estimate outcome regression models
        # Outcome model for treated
        treated_data = data[data[treatment_col] == 1]
        if len(treated_data) > 0:
            X_treated = treated_data[confounder_cols].fillna(treated_data[confounder_cols].median())
            self.outcome_model_treated = RandomForestRegressor(random_state=42)
            self.outcome_model_treated.fit(X_treated, treated_data[outcome_col])

        # Outcome model for control
        control_data = data[data[treatment_col] == 0]
        if len(control_data) > 0:
            X_control = control_data[confounder_cols].fillna(control_data[confounder_cols].median())
            self.outcome_model_control = RandomForestRegressor(random_state=42)
            self.outcome_model_control.fit(X_control, control_data[outcome_col])

        # Step 3: Predict potential outcomes for all units
        X_all = data[confounder_cols].fillna(data[confounder_cols].median())

        if len(treated_data) > 0:
            mu1 = self.outcome_model_treated.predict(X_all)  # E[Y(1)|X]
        else:
            mu1 = np.zeros(len(data))

        if len(control_data) > 0:
            mu0 = self.outcome_model_control.predict(X_all)  # E[Y(0)|X]
        else:
            mu0 = np.zeros(len(data))

        # Step 4: Doubly robust estimator
        T = data[treatment_col].values
        Y = data[outcome_col].values

        # IPW terms
        ipw_1 = T * Y / ps
        ipw_0 = (1 - T) * Y / (1 - ps)

        # Regression adjustment terms
        reg_adj_1 = (T - ps) * mu1 / ps
        reg_adj_0 = (T - ps) * mu0 / (1 - ps)

        # Doubly robust estimates
        mu1_dr = ipw_1 + reg_adj_1
        mu0_dr = ipw_0 - reg_adj_0

        # Average treatment effect
        ate = np.mean(mu1_dr) - np.mean(mu0_dr)

        # Standard error (simplified)
        influence_function = (mu1_dr - mu0_dr) - ate
        se = np.sqrt(np.var(influence_function) / len(data))

        # Confidence interval and p-value
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        p_value = 2 * (1 - stats.norm.cdf(abs(ate / se)))

        return CausalEstimate(
            treatment=treatment_col,
            outcome=outcome_col,
            effect_size=ate,
            standard_error=se,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method="doubly_robust",
            sample_size=len(data),
            confounders_adjusted=confounder_cols
        )


class CausalInferenceFramework:
    """
    Comprehensive framework for causal inference in stem cell therapy research
    """

    def __init__(self):
        self.dag = CausalDAG()
        self.propensity_analysis = PropensityScoreAnalysis()
        self.iv_analysis = InstrumentalVariableAnalysis()
        self.mediation_analysis = MediationAnalysis()
        self.doubly_robust = DoublyRobustEstimation()

        self.results = {}

    def build_stem_cell_dag(self) -> CausalDAG:
        """Build a causal DAG for stem cell therapy analysis"""

        # Add variables
        variables = {
            'cell_dose': ('treatment', 'Cell dose administered'),
            'injection_route': ('treatment', 'Route of administration'),
            'patient_age': ('confounder', 'Patient age'),
            'disease_severity': ('confounder', 'Baseline disease severity'),
            'comorbidities': ('confounder', 'Patient comorbidities'),
            'previous_treatments': ('confounder', 'Previous treatment history'),
            'hospital_quality': ('confounder', 'Hospital quality/experience'),
            'cell_viability': ('mediator', 'Cell viability at injection'),
            'immune_response': ('mediator', 'Patient immune response'),
            'engraftment': ('mediator', 'Cell engraftment success'),
            'primary_outcome': ('outcome', 'Primary efficacy outcome'),
            'adverse_events': ('outcome', 'Safety outcomes'),
            'quality_of_life': ('outcome', 'Quality of life measures')
        }

        for var_name, (var_type, description) in variables.items():
            self.dag.add_variable(var_name, var_type, description)

        # Add causal relationships
        causal_edges = [
            # Treatment effects
            ('cell_dose', 'cell_viability'),
            ('cell_dose', 'immune_response'),
            ('cell_dose', 'adverse_events'),
            ('injection_route', 'engraftment'),
            ('injection_route', 'adverse_events'),

            # Patient characteristics effects
            ('patient_age', 'immune_response'),
            ('patient_age', 'adverse_events'),
            ('disease_severity', 'primary_outcome'),
            ('disease_severity', 'quality_of_life'),
            ('comorbidities', 'adverse_events'),
            ('comorbidities', 'primary_outcome'),

            # Previous treatments
            ('previous_treatments', 'immune_response'),
            ('previous_treatments', 'primary_outcome'),

            # Hospital quality
            ('hospital_quality', 'cell_viability'),
            ('hospital_quality', 'adverse_events'),

            # Mediation pathways
            ('cell_viability', 'engraftment'),
            ('immune_response', 'engraftment'),
            ('engraftment', 'primary_outcome'),
            ('immune_response', 'adverse_events'),

            # Outcome relationships
            ('adverse_events', 'quality_of_life'),
            ('primary_outcome', 'quality_of_life')
        ]

        for cause, effect in causal_edges:
            self.dag.add_causal_edge(cause, effect)

        return self.dag

    def comprehensive_causal_analysis(self, data: pd.DataFrame,
                                    treatment_col: str, outcome_col: str,
                                    confounder_cols: List[str] = None,
                                    instrument_col: str = None,
                                    mediator_col: str = None) -> Dict[str, Any]:
        """Perform comprehensive causal analysis using multiple methods"""

        print("Performing comprehensive causal inference analysis...")

        results = {}

        # 1. Propensity Score Analysis
        if confounder_cols:
            print("1. Propensity Score Analysis")

            # Matching
            try:
                matched_data = self.propensity_analysis.propensity_score_matching(
                    data, treatment_col, outcome_col, confounder_cols
                )
                if len(matched_data) > 0:
                    matching_result = self.propensity_analysis.estimate_treatment_effect_matching(
                        outcome_col, treatment_col
                    )
                    results['propensity_matching'] = matching_result
            except Exception as e:
                print(f"Propensity score matching failed: {e}")

            # IPW
            try:
                ipw_result = self.propensity_analysis.ipw_estimation(
                    data, treatment_col, outcome_col, confounder_cols
                )
                results['inverse_probability_weighting'] = ipw_result
            except Exception as e:
                print(f"IPW estimation failed: {e}")

        # 2. Instrumental Variable Analysis
        if instrument_col and instrument_col in data.columns:
            print("2. Instrumental Variable Analysis")

            try:
                # Validate instrument
                iv_validation = self.iv_analysis.validate_instrument(
                    data, treatment_col, outcome_col, instrument_col
                )
                results['instrument_validation'] = iv_validation

                # 2SLS estimation
                iv_result = self.iv_analysis.two_stage_least_squares(
                    data, treatment_col, outcome_col, instrument_col, confounder_cols
                )
                results['instrumental_variable'] = iv_result
            except Exception as e:
                print(f"IV analysis failed: {e}")

        # 3. Mediation Analysis
        if mediator_col and mediator_col in data.columns:
            print("3. Mediation Analysis")

            try:
                mediation_result = self.mediation_analysis.causal_mediation_analysis(
                    data, treatment_col, mediator_col, outcome_col, confounder_cols
                )
                results['mediation_analysis'] = mediation_result
            except Exception as e:
                print(f"Mediation analysis failed: {e}")

        # 4. Doubly Robust Estimation
        if confounder_cols:
            print("4. Doubly Robust Estimation")

            try:
                dr_result = self.doubly_robust.doubly_robust_ate(
                    data, treatment_col, outcome_col, confounder_cols
                )
                results['doubly_robust'] = dr_result
            except Exception as e:
                print(f"Doubly robust estimation failed: {e}")

        # 5. Simple regression for comparison
        print("5. Naive Regression (for comparison)")
        try:
            X = data[confounder_cols] if confounder_cols else pd.DataFrame(index=data.index)
            X[treatment_col] = data[treatment_col]
            X = sm.add_constant(X)

            naive_model = sm.OLS(data[outcome_col], X).fit()
            treatment_effect = naive_model.params[treatment_col]
            se = naive_model.bse[treatment_col]

            naive_result = CausalEstimate(
                treatment=treatment_col,
                outcome=outcome_col,
                effect_size=treatment_effect,
                standard_error=se,
                confidence_interval=(treatment_effect - 1.96*se, treatment_effect + 1.96*se),
                p_value=naive_model.pvalues[treatment_col],
                method="naive_regression",
                sample_size=len(data),
                confounders_adjusted=confounder_cols or []
            )
            results['naive_regression'] = naive_result

        except Exception as e:
            print(f"Naive regression failed: {e}")

        self.results = results
        return results

    def compare_methods(self) -> pd.DataFrame:
        """Compare results across different causal inference methods"""

        if not self.results:
            raise ValueError("Must run causal analysis first")

        comparison_data = []

        for method, result in self.results.items():
            if isinstance(result, CausalEstimate):
                comparison_data.append({
                    'Method': method,
                    'Effect_Size': result.effect_size,
                    'Standard_Error': result.standard_error,
                    'P_Value': result.p_value,
                    'CI_Lower': result.confidence_interval[0],
                    'CI_Upper': result.confidence_interval[1],
                    'Sample_Size': result.sample_size,
                    'Significant': result.p_value < 0.05
                })

        return pd.DataFrame(comparison_data)


def main():
    """Main function to demonstrate causal inference capabilities"""

    print("="*60)
    print("CAUSAL INFERENCE FRAMEWORK FOR STEM CELL THERAPY")
    print("="*60)

    # Load and prepare data
    try:
        data = pd.read_csv('../data/clinical_trial_data.csv')
        print(f"Loaded {len(data)} clinical trials")
    except:
        # Create synthetic data for demonstration
        print("Creating synthetic data for demonstration...")
        np.random.seed(42)
        n = 500

        data = pd.DataFrame({
            'patient_age': np.random.normal(55, 12, n),
            'disease_severity': np.random.beta(3, 3, n),
            'comorbidities': np.random.binomial(1, 0.3, n),
            'previous_treatments': np.random.poisson(1, n),
            'hospital_quality': np.random.beta(5, 2, n),
        })

        # Treatment assignment (with confounding)
        treatment_prob = 0.3 + 0.2 * (data['hospital_quality'] > 0.7) - 0.1 * (data['patient_age'] > 65)
        data['cell_dose_high'] = np.random.binomial(1, treatment_prob, n)

        # Mediator
        data['engraftment_success'] = np.random.beta(
            2 + 3 * data['cell_dose_high'] + 2 * data['hospital_quality'],
            3 - data['cell_dose_high']
        )

        # Outcome (with treatment effect and confounding)
        outcome_mean = (50 + 20 * data['cell_dose_high'] + 15 * data['engraftment_success'] -
                       10 * data['disease_severity'] - 5 * (data['patient_age'] > 65))
        data['primary_outcome'] = np.random.normal(outcome_mean, 10)

        # Instrument (hospital assignment - affects treatment but not outcome directly)
        data['hospital_id'] = np.random.choice(['A', 'B', 'C'], n, p=[0.4, 0.3, 0.3])
        data['hospital_assignment'] = (data['hospital_id'] == 'A').astype(int)

    # Initialize framework
    framework = CausalInferenceFramework()

    # Build causal DAG
    print("\n1. BUILDING CAUSAL DAG")
    print("-" * 30)
    dag = framework.build_stem_cell_dag()
    print(f"DAG constructed with {len(dag.graph.nodes)} variables and {len(dag.graph.edges)} causal edges")

    # Identify confounders
    if 'cell_dose_high' in data.columns and 'primary_outcome' in data.columns:
        confounders = dag.get_confounders('cell_dose', 'primary_outcome')
        print(f"Identified confounders: {confounders}")

    # Comprehensive causal analysis
    print("\n2. COMPREHENSIVE CAUSAL ANALYSIS")
    print("-" * 40)

    confounder_cols = ['patient_age', 'disease_severity', 'comorbidities', 'previous_treatments', 'hospital_quality']
    # Filter to existing columns
    confounder_cols = [col for col in confounder_cols if col in data.columns]

    results = framework.comprehensive_causal_analysis(
        data=data,
        treatment_col='cell_dose_high',
        outcome_col='primary_outcome',
        confounder_cols=confounder_cols,
        instrument_col='hospital_assignment' if 'hospital_assignment' in data.columns else None,
        mediator_col='engraftment_success' if 'engraftment_success' in data.columns else None
    )

    # Compare methods
    print("\n3. METHOD COMPARISON")
    print("-" * 30)

    try:
        comparison_df = framework.compare_methods()
        print(comparison_df.round(4))

        # Summary statistics
        print(f"\nEffect size range: {comparison_df['Effect_Size'].min():.3f} to {comparison_df['Effect_Size'].max():.3f}")
        print(f"Methods with significant effects: {comparison_df['Significant'].sum()}/{len(comparison_df)}")

    except Exception as e:
        print(f"Method comparison failed: {e}")

    print("\n" + "="*60)
    print("CAUSAL INFERENCE ANALYSIS COMPLETED!")
    print("="*60)

    return results


if __name__ == "__main__":
    results = main()