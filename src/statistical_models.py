"""
Statistical Models for Stem Cell Therapy Analysis

This module implements advanced statistical models for predicting treatment outcomes
and analyzing clinical trial data for stem cell therapy in epilepsy and diabetes.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import pymc as pm
import arviz as az
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SurvivalAnalysisModel:
    """Survival analysis for treatment durability and time-to-event modeling"""

    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter()
        self.survival_data = None

    def prepare_survival_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare survival data from clinical trial dataframe

        Args:
            df: Clinical trial dataframe

        Returns:
            Survival analysis ready dataframe
        """
        # Create survival dataset
        survival_records = []

        for _, row in df.iterrows():
            if pd.notna(row['follow_up_months']) and pd.notna(row['endpoint_value']):
                # Define event (treatment failure) based on efficacy threshold
                if row['primary_endpoint'] in ['seizure_reduction_percent', 'insulin_independence_rate']:
                    # Treatment failure if efficacy < 50%
                    event = 1 if row['endpoint_value'] < 50 else 0
                else:
                    # For continuous outcomes, define failure as lack of improvement
                    event = 1 if row['endpoint_value'] <= 0 else 0

                survival_records.append({
                    'duration': row['follow_up_months'],
                    'event': event,
                    'condition': row['condition'],
                    'intervention': row['intervention'],
                    'n_patients': row['n_patients'],
                    'efficacy': row['endpoint_value']
                })

        self.survival_data = pd.DataFrame(survival_records)
        return self.survival_data

    def kaplan_meier_analysis(self, group_by: str = 'condition') -> dict:
        """
        Perform Kaplan-Meier survival analysis

        Args:
            group_by: Column to group survival curves by

        Returns:
            Dictionary with survival analysis results
        """
        if self.survival_data is None:
            raise ValueError("Must prepare survival data first")

        results = {}

        # Overall survival curve
        self.kmf.fit(self.survival_data['duration'],
                    self.survival_data['event'],
                    label='Overall')

        results['overall'] = {
            'median_survival': self.kmf.median_survival_time_,
            'survival_function': self.kmf.survival_function_,
            'confidence_interval': self.kmf.confidence_interval_
        }

        # Group-specific survival curves
        groups = self.survival_data[group_by].unique()
        results['groups'] = {}

        for group in groups:
            if pd.notna(group):
                group_data = self.survival_data[self.survival_data[group_by] == group]

                kmf_group = KaplanMeierFitter()
                kmf_group.fit(group_data['duration'],
                             group_data['event'],
                             label=str(group))

                results['groups'][group] = {
                    'median_survival': kmf_group.median_survival_time_,
                    'survival_function': kmf_group.survival_function_,
                    'n_patients': len(group_data)
                }

        # Log-rank test for group comparison
        if len(groups) == 2:
            group1_data = self.survival_data[self.survival_data[group_by] == groups[0]]
            group2_data = self.survival_data[self.survival_data[group_by] == groups[1]]

            logrank_result = logrank_test(
                group1_data['duration'], group2_data['duration'],
                group1_data['event'], group2_data['event']
            )

            results['logrank_test'] = {
                'test_statistic': logrank_result.test_statistic,
                'p_value': logrank_result.p_value,
                'is_significant': logrank_result.p_value < 0.05
            }

        return results

    def cox_regression_analysis(self, covariates: list) -> dict:
        """
        Perform Cox proportional hazards regression

        Args:
            covariates: List of covariate column names

        Returns:
            Cox regression results
        """
        if self.survival_data is None:
            raise ValueError("Must prepare survival data first")

        # Prepare data for Cox regression
        cox_data = self.survival_data.copy()

        # Add dummy variables for categorical covariates
        for covariate in covariates:
            if cox_data[covariate].dtype == 'object':
                dummies = pd.get_dummies(cox_data[covariate], prefix=covariate)
                cox_data = pd.concat([cox_data, dummies], axis=1)
                cox_data.drop(covariate, axis=1, inplace=True)

        # Fit Cox model
        self.cph.fit(cox_data, duration_col='duration', event_col='event')

        return {
            'summary': self.cph.summary,
            'hazard_ratios': self.cph.hazard_ratios_,
            'confidence_intervals': self.cph.confidence_intervals_,
            'p_values': self.cph.summary['p'],
            'concordance_index': self.cph.concordance_index_
        }


class BayesianMetaAnalysis:
    """Bayesian meta-analysis for combining trial results"""

    def __init__(self):
        self.model = None
        self.trace = None

    def hierarchical_meta_analysis(self, effect_sizes: np.ndarray,
                                  standard_errors: np.ndarray,
                                  study_names: list = None) -> dict:
        """
        Perform Bayesian hierarchical meta-analysis

        Args:
            effect_sizes: Array of study effect sizes
            standard_errors: Array of study standard errors
            study_names: List of study identifiers

        Returns:
            Meta-analysis results
        """
        n_studies = len(effect_sizes)

        with pm.Model() as model:
            # Hyperpriors for population-level parameters
            mu = pm.Normal('mu', mu=0, sigma=10)  # Population mean effect
            tau = pm.HalfCauchy('tau', beta=2.5)  # Between-study heterogeneity

            # Study-specific effects
            theta = pm.Normal('theta', mu=mu, sigma=tau, shape=n_studies)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=theta, sigma=standard_errors,
                            observed=effect_sizes)

            # Sample from posterior
            trace = pm.sample(2000, tune=1000, random_seed=42,
                            target_accept=0.95, return_inferencedata=True)

        self.model = model
        self.trace = trace

        # Extract results
        posterior_mu = trace.posterior['mu'].values.flatten()
        posterior_tau = trace.posterior['tau'].values.flatten()
        posterior_theta = trace.posterior['theta'].values

        results = {
            'population_effect': {
                'mean': np.mean(posterior_mu),
                'median': np.median(posterior_mu),
                'std': np.std(posterior_mu),
                'hdi_95': az.hdi(trace, var_names=['mu'], hdi_prob=0.95)['mu']
            },
            'heterogeneity': {
                'tau_mean': np.mean(posterior_tau),
                'tau_median': np.median(posterior_tau),
                'tau_hdi_95': az.hdi(trace, var_names=['tau'], hdi_prob=0.95)['tau']
            },
            'study_effects': {},
            'model_diagnostics': {
                'r_hat': az.rhat(trace),
                'ess_bulk': az.ess(trace, method='bulk'),
                'ess_tail': az.ess(trace, method='tail')
            }
        }

        # Study-specific results
        for i in range(n_studies):
            study_name = study_names[i] if study_names else f'Study_{i+1}'
            study_posterior = posterior_theta[:, :, i].flatten()

            results['study_effects'][study_name] = {
                'mean': np.mean(study_posterior),
                'median': np.median(study_posterior),
                'std': np.std(study_posterior),
                'hdi_95': az.hdi(trace.posterior['theta'].sel(theta_dim_0=i),
                               hdi_prob=0.95).values
            }

        return results

    def prediction_interval(self, alpha: float = 0.05) -> tuple:
        """
        Calculate prediction interval for future studies

        Args:
            alpha: Significance level (default 0.05 for 95% interval)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.trace is None:
            raise ValueError("Must run meta-analysis first")

        # Generate predictions for a new study
        posterior_mu = self.trace.posterior['mu'].values.flatten()
        posterior_tau = self.trace.posterior['tau'].values.flatten()

        # Prediction distribution
        predictions = []
        for mu, tau in zip(posterior_mu, posterior_tau):
            pred = np.random.normal(mu, tau)
            predictions.append(pred)

        predictions = np.array(predictions)

        lower = np.percentile(predictions, (alpha/2) * 100)
        upper = np.percentile(predictions, (1 - alpha/2) * 100)

        return lower, upper


class TreatmentResponsePredictor:
    """Machine learning models for predicting treatment response"""

    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.fitted_models = {}
        self.feature_importance = {}

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for machine learning

        Args:
            df: Clinical trial dataframe

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Create feature matrix
        features = []
        targets = []
        feature_names = []

        for _, row in df.iterrows():
            if pd.notna(row['endpoint_value']) and pd.notna(row['n_patients']):
                # Numerical features
                feature_vector = [
                    row['n_patients'],
                    row['follow_up_months'] if pd.notna(row['follow_up_months']) else 12,
                    row['safety_events'] if pd.notna(row['safety_events']) else 0
                ]

                # Categorical features (one-hot encoded)
                condition_features = [
                    1 if row['condition'] == 'Epilepsy' else 0,
                    1 if row['condition'] == 'Type_1_Diabetes' else 0,
                    1 if row['condition'] == 'Type_2_Diabetes' else 0
                ]

                intervention_features = [
                    1 if 'MSC' in str(row['intervention']) else 0,
                    1 if 'NRTX' in str(row['intervention']) else 0,
                    1 if 'VX-880' in str(row['intervention']) else 0
                ]

                phase_features = [
                    1 if row['phase'] == 'Phase_1_2' else 0,
                    1 if row['phase'] == 'Phase_1_2_3' else 0,
                    1 if row['phase'] == 'Meta_Analysis' else 0
                ]

                feature_vector.extend(condition_features)
                feature_vector.extend(intervention_features)
                feature_vector.extend(phase_features)

                features.append(feature_vector)
                targets.append(row['endpoint_value'])

        feature_names = [
            'n_patients', 'follow_up_months', 'safety_events',
            'condition_epilepsy', 'condition_t1dm', 'condition_t2dm',
            'intervention_msc', 'intervention_nrtx', 'intervention_vx880',
            'phase_1_2', 'phase_1_2_3', 'phase_meta'
        ]

        return np.array(features), np.array(targets), feature_names

    def train_models(self, X: np.ndarray, y: np.ndarray,
                    feature_names: list) -> dict:
        """
        Train machine learning models

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names

        Returns:
            Training results dictionary
        """
        results = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        for model_name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            self.fitted_models[model_name] = model

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5,
                                      scoring='r2')

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[model_name] = importance

            results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': self.feature_importance.get(model_name, {})
            }

        return results

    def predict_treatment_response(self, patient_features: dict,
                                 model_name: str = 'random_forest') -> dict:
        """
        Predict treatment response for a new patient

        Args:
            patient_features: Dictionary of patient characteristics
            model_name: Name of model to use for prediction

        Returns:
            Prediction results
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.fitted_models[model_name]

        # Convert patient features to feature vector
        feature_vector = [
            patient_features.get('n_patients', 50),
            patient_features.get('follow_up_months', 12),
            patient_features.get('safety_events', 0),
            1 if patient_features.get('condition') == 'Epilepsy' else 0,
            1 if patient_features.get('condition') == 'Type_1_Diabetes' else 0,
            1 if patient_features.get('condition') == 'Type_2_Diabetes' else 0,
            1 if 'MSC' in patient_features.get('intervention', '') else 0,
            1 if 'NRTX' in patient_features.get('intervention', '') else 0,
            1 if 'VX-880' in patient_features.get('intervention', '') else 0,
            1 if patient_features.get('phase') == 'Phase_1_2' else 0,
            1 if patient_features.get('phase') == 'Phase_1_2_3' else 0,
            1 if patient_features.get('phase') == 'Meta_Analysis' else 0
        ]

        # Make prediction
        prediction = model.predict([feature_vector])[0]

        # Calculate prediction interval (for tree-based models)
        if hasattr(model, 'estimators_'):
            predictions = [tree.predict([feature_vector])[0]
                         for tree in model.estimators_]
            prediction_std = np.std(predictions)

            return {
                'predicted_response': prediction,
                'prediction_std': prediction_std,
                'confidence_interval_95': [
                    prediction - 1.96 * prediction_std,
                    prediction + 1.96 * prediction_std
                ]
            }
        else:
            return {
                'predicted_response': prediction
            }


def meta_analysis_diabetes_hba1c(df: pd.DataFrame) -> dict:
    """
    Perform meta-analysis of diabetes HbA1c outcomes

    Args:
        df: Clinical trial dataframe

    Returns:
        Meta-analysis results
    """
    # Filter diabetes trials with HbA1c data
    hba1c_data = df[
        (df['condition'].str.contains('Diabetes', na=False)) &
        (df['primary_endpoint'] == 'hba1c_change_absolute') &
        (df['ci_lower'].notna()) &
        (df['ci_upper'].notna())
    ].copy()

    if len(hba1c_data) < 2:
        return {"error": "Insufficient data for meta-analysis"}

    # Extract effect sizes and standard errors
    effect_sizes = hba1c_data['endpoint_value'].values
    standard_errors = ((hba1c_data['ci_upper'] - hba1c_data['ci_lower']) / (2 * 1.96)).values
    study_names = hba1c_data['trial_id'].values

    # Perform Bayesian meta-analysis
    meta_analyzer = BayesianMetaAnalysis()
    results = meta_analyzer.hierarchical_meta_analysis(
        effect_sizes, standard_errors, study_names
    )

    # Add prediction interval
    pred_lower, pred_upper = meta_analyzer.prediction_interval()
    results['prediction_interval_95'] = [pred_lower, pred_upper]

    return results


def survival_analysis_treatment_durability(df: pd.DataFrame) -> dict:
    """
    Analyze treatment durability using survival analysis

    Args:
        df: Clinical trial dataframe

    Returns:
        Survival analysis results
    """
    survival_model = SurvivalAnalysisModel()
    survival_data = survival_model.prepare_survival_data(df)

    if len(survival_data) < 5:
        return {"error": "Insufficient data for survival analysis"}

    # Kaplan-Meier analysis
    km_results = survival_model.kaplan_meier_analysis('condition')

    # Cox regression (if enough covariates)
    available_covariates = ['condition', 'n_patients', 'efficacy']
    cox_results = None

    try:
        cox_results = survival_model.cox_regression_analysis(available_covariates)
    except Exception as e:
        cox_results = {"error": f"Cox regression failed: {str(e)}"}

    return {
        'kaplan_meier': km_results,
        'cox_regression': cox_results,
        'survival_data_summary': {
            'n_records': len(survival_data),
            'n_events': survival_data['event'].sum(),
            'median_followup': survival_data['duration'].median()
        }
    }


def main():
    """Main function to demonstrate statistical modeling"""
    print("Loading clinical trial data...")
    df = pd.read_csv('../data/clinical_trial_data.csv')

    print("\n=== BAYESIAN META-ANALYSIS ===")
    meta_results = meta_analysis_diabetes_hba1c(df)
    if 'error' not in meta_results:
        print(f"Population effect: {meta_results['population_effect']['mean']:.2f} ± {meta_results['population_effect']['std']:.2f}")
        print(f"95% HDI: {meta_results['population_effect']['hdi_95']}")
    else:
        print(meta_results['error'])

    print("\n=== SURVIVAL ANALYSIS ===")
    survival_results = survival_analysis_treatment_durability(df)
    if 'error' not in survival_results:
        print(f"Survival data records: {survival_results['survival_data_summary']['n_records']}")
        print(f"Events observed: {survival_results['survival_data_summary']['n_events']}")
    else:
        print(survival_results['error'])

    print("\n=== MACHINE LEARNING PREDICTION ===")
    predictor = TreatmentResponsePredictor()
    X, y, feature_names = predictor.prepare_features(df)

    if len(X) > 5:
        ml_results = predictor.train_models(X, y, feature_names)

        for model_name, results in ml_results.items():
            print(f"{model_name.upper()}:")
            print(f"  Test R²: {results['test_r2']:.3f}")
            print(f"  CV Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    else:
        print("Insufficient data for machine learning")


if __name__ == "__main__":
    main()