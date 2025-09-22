"""
Advanced Predictive Modeling Framework for Stem Cell Therapy Analysis

This module implements sophisticated machine learning and statistical models
to predict treatment outcomes, patient responses, and optimal protocols.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                            ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                BayesianRidge, HuberRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                   train_test_split, KFold, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           explained_variance_score, median_absolute_error)
from sklearn.feature_selection import (SelectKBest, f_regression, RFE,
                                     SelectFromModel)
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
from scipy.optimize import minimize
import optuna
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class TreatmentOutcomePredictor:
    """
    Advanced predictor for stem cell therapy treatment outcomes
    """

    def __init__(self, target_variable: str = 'endpoint_value'):
        self.target_variable = target_variable
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.trained_models = {}
        self.performance_metrics = {}
        self.feature_importance = {}

    def initialize_models(self) -> None:
        """Initialize all prediction models"""
        self.models = {
            # Linear Models
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'bayesian_ridge': BayesianRidge(),
            'huber': HuberRegressor(),

            # Tree-based Models
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ada_boost': AdaBoostRegressor(n_estimators=100, random_state=42),

            # Advanced Boosting
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),

            # Support Vector Machines
            'svr_linear': SVR(kernel='linear'),
            'svr_rbf': SVR(kernel='rbf'),
            'svr_poly': SVR(kernel='poly', degree=3),

            # Neural Networks
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for machine learning with advanced engineering
        """
        # Identify feature columns
        exclude_cols = [self.target_variable, 'trial_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Create feature matrix
        X = df[feature_cols].copy()
        y = df[self.target_variable].copy()

        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Label encoding for categorical variables
            X[col] = pd.Categorical(X[col]).codes

        # Feature engineering
        X_engineered = self._engineer_features(X)

        return X_engineered.values, y.values, list(X_engineered.columns)

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        X_eng = X.copy()

        # Polynomial features for numerical columns
        numerical_cols = X_eng.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) >= 2:
            # Add interaction terms for important pairs
            important_pairs = [
                ('n_patients', 'follow_up_months'),
                ('endpoint_value', 'safety_events'),
                ('treatment_group', 'control_group')
            ]

            for col1, col2 in important_pairs:
                if col1 in X_eng.columns and col2 in X_eng.columns:
                    X_eng[f'{col1}_x_{col2}'] = X_eng[col1] * X_eng[col2]
                    X_eng[f'{col1}_div_{col2}'] = X_eng[col1] / (X_eng[col2] + 1e-8)

        # Ratio features
        if 'treatment_group' in X_eng.columns and 'control_group' in X_eng.columns:
            X_eng['treatment_ratio'] = X_eng['treatment_group'] / (X_eng['treatment_group'] + X_eng['control_group'] + 1e-8)

        # Log transformations for skewed features
        for col in numerical_cols:
            if col in X_eng.columns and X_eng[col].min() > 0:
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])

        # Squared terms for potential non-linear relationships
        for col in ['follow_up_months', 'n_patients']:
            if col in X_eng.columns:
                X_eng[f'{col}_squared'] = X_eng[col] ** 2

        return X_eng

    def _initialize_ensemble_models(self) -> None:
        """Initialize ensemble models"""
        # Base estimators for ensemble
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42)),
            ('svr', SVR(kernel='rbf')),
            ('ridge', Ridge(alpha=1.0))
        ]

        self.ensemble_models = {
            # Voting Ensemble
            'voting_regressor': VotingRegressor(estimators=base_models),

            # Bagging Ensemble
            'bagging': BaggingRegressor(
                base_estimator=DecisionTreeRegressor(random_state=42),
                n_estimators=50, random_state=42
            ),

            # Stacking Ensemble
            'stacking': StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )
        }

    def feature_selection(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str], method: str = 'all') -> Tuple[np.ndarray, List[str]]:
        """
        Advanced feature selection using multiple methods
        """
        if method == 'all':
            # Combine multiple feature selection methods
            selected_features = set(range(X.shape[1]))

            # 1. Univariate feature selection
            k_best = min(20, X.shape[1])
            selector_univariate = SelectKBest(score_func=f_regression, k=k_best)
            selector_univariate.fit(X, y)
            univariate_features = set(selector_univariate.get_support(indices=True))

            # 2. Recursive Feature Elimination
            if X.shape[1] > 5:
                rfe_features = min(15, X.shape[1])
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector_rfe = RFE(estimator, n_features_to_select=rfe_features)
                selector_rfe.fit(X, y)
                rfe_selected = set(selector_rfe.get_support(indices=True))
            else:
                rfe_selected = selected_features

            # 3. Model-based feature selection
            estimator_lasso = Lasso(alpha=0.1)
            try:
                estimator_lasso.fit(X, y)
                selector_model = SelectFromModel(estimator_lasso, prefit=True)
                model_selected = set(selector_model.get_support(indices=True))
            except:
                model_selected = selected_features

            # Combine selections (intersection for conservative approach)
            if len(univariate_features & rfe_selected & model_selected) >= 3:
                final_features = univariate_features & rfe_selected & model_selected
            else:
                # Union if intersection is too small
                final_features = univariate_features | rfe_selected | model_selected

            final_features = list(final_features)

        else:
            final_features = list(range(X.shape[1]))

        selected_feature_names = [feature_names[i] for i in final_features]

        self.feature_selectors['selected_indices'] = final_features
        self.feature_selectors['selected_names'] = selected_feature_names

        return X[:, final_features], selected_feature_names

    def hyperparameter_optimization(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize hyperparameters using Optuna
        """
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)

            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)

            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)

            elif model_name == 'svr_rbf':
                params = {
                    'C': trial.suggest_float('C', 0.1, 100, log=True),
                    'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
                }
                model = SVR(kernel='rbf', **params)

            else:
                # Use default model
                model = self.models[model_name]

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            return np.mean(cv_scores)

        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        return study.best_params

    def train_all_models(self, df: pd.DataFrame, optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train all models with comprehensive evaluation
        """
        # Initialize models
        self.initialize_models()

        # Prepare data
        X, y, feature_names = self.prepare_features(df)

        if len(X) < 5:
            return {'error': 'Insufficient data for training'}

        # Feature selection
        X_selected, selected_features = self.feature_selection(X, y, feature_names)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['feature_scaler'] = scaler

        results = {}

        # Train each model
        for model_name, model in self.models.items():
            try:
                print(f"Training {model_name}...")

                # Hyperparameter optimization for selected models
                if optimize_hyperparams and model_name in ['random_forest', 'xgboost', 'lightgbm', 'svr_rbf']:
                    best_params = self.hyperparameter_optimization(model_name, X_train_scaled, y_train)

                    # Update model with best parameters
                    if model_name == 'random_forest':
                        model = RandomForestRegressor(**best_params)
                    elif model_name == 'xgboost':
                        model = xgb.XGBRegressor(**best_params)
                    elif model_name == 'lightgbm':
                        model = lgb.LGBMRegressor(**best_params)
                    elif model_name == 'svr_rbf':
                        model = SVR(kernel='rbf', **best_params)

                # Train model
                if model_name.startswith('svr') or model_name == 'mlp':
                    # Use scaled features for SVM and neural networks
                    model.fit(X_train_scaled, y_train)
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                else:
                    # Use original features for tree-based models
                    model.fit(X_train, y_train)
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                # Calculate performance metrics
                metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)

                # Cross-validation scores
                if model_name.startswith('svr') or model_name == 'mlp':
                    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')
                else:
                    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')

                metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
                metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())

                # Feature importance (if available)
                feature_importance = self._extract_feature_importance(model, selected_features)

                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': feature_importance,
                    'hyperparameters': best_params if optimize_hyperparams and model_name in ['random_forest', 'xgboost', 'lightgbm', 'svr_rbf'] else None
                }

                self.trained_models[model_name] = model
                self.performance_metrics[model_name] = metrics
                self.feature_importance[model_name] = feature_importance

            except Exception as e:
                print(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        # Model ranking
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        if valid_models:
            model_ranking = self._rank_models(valid_models)
            results['model_ranking'] = model_ranking

        results['selected_features'] = selected_features
        results['data_info'] = {
            'n_samples': len(X),
            'n_features_original': len(feature_names),
            'n_features_selected': len(selected_features),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return results

    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Training metrics
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        metrics['train_r2'] = r2_score(y_train, y_pred_train)
        metrics['train_explained_variance'] = explained_variance_score(y_train, y_pred_train)

        # Test metrics
        metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
        metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
        metrics['test_r2'] = r2_score(y_test, y_pred_test)
        metrics['test_explained_variance'] = explained_variance_score(y_test, y_pred_test)

        # Overfitting indicator
        metrics['overfitting_ratio'] = metrics['test_rmse'] / metrics['train_rmse']

        # Additional metrics
        metrics['test_median_ae'] = median_absolute_error(y_test, y_pred_test)

        return metrics

    def _extract_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        importance_dict = {}

        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))

            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
                importance_dict = dict(zip(feature_names, importances))

            elif hasattr(model, 'support_'):
                # Feature selection models
                selected_features = model.support_
                importance_dict = {name: 1.0 if selected else 0.0
                                 for name, selected in zip(feature_names, selected_features)}

        except Exception:
            # Return empty dict if extraction fails
            pass

        return importance_dict

    def _rank_models(self, model_results: Dict) -> List[Dict]:
        """Rank models based on multiple criteria"""
        rankings = []

        for model_name, results in model_results.items():
            if 'metrics' in results:
                metrics = results['metrics']

                # Create composite score (lower is better for RMSE, higher for R²)
                composite_score = (
                    -metrics.get('test_rmse', float('inf')) * 0.4 +  # Negative because lower is better
                    metrics.get('test_r2', 0) * 0.4 +
                    -metrics.get('overfitting_ratio', float('inf')) * 0.2
                )

                rankings.append({
                    'model_name': model_name,
                    'test_rmse': metrics.get('test_rmse', float('inf')),
                    'test_r2': metrics.get('test_r2', 0),
                    'overfitting_ratio': metrics.get('overfitting_ratio', float('inf')),
                    'composite_score': composite_score
                })

        # Sort by composite score (descending)
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)

        return rankings

    def predict_treatment_outcome(self, patient_features: Dict,
                                model_name: str = None) -> Dict[str, Any]:
        """
        Predict treatment outcome for new patient
        """
        if not self.trained_models:
            return {'error': 'No trained models available'}

        # Use best model if not specified
        if model_name is None:
            if 'model_ranking' in self.performance_metrics:
                model_name = self.performance_metrics['model_ranking'][0]['model_name']
            else:
                model_name = list(self.trained_models.keys())[0]

        if model_name not in self.trained_models:
            return {'error': f'Model {model_name} not available'}

        try:
            # Convert patient features to feature vector
            feature_vector = self._patient_features_to_vector(patient_features)

            # Apply feature selection
            if 'selected_indices' in self.feature_selectors:
                feature_vector = feature_vector[self.feature_selectors['selected_indices']]

            # Scale features if needed
            if model_name.startswith('svr') or model_name == 'mlp':
                if 'feature_scaler' in self.scalers:
                    feature_vector = self.scalers['feature_scaler'].transform([feature_vector])[0]

            # Make prediction
            model = self.trained_models[model_name]
            prediction = model.predict([feature_vector])[0]

            # Calculate prediction interval (for ensemble models)
            prediction_interval = self._calculate_prediction_interval(
                model, feature_vector, model_name
            )

            return {
                'predicted_outcome': prediction,
                'model_used': model_name,
                'prediction_interval': prediction_interval,
                'model_performance': self.performance_metrics.get(model_name, {}),
                'feature_importance': self.feature_importance.get(model_name, {})
            }

        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

    def _patient_features_to_vector(self, patient_features: Dict) -> np.ndarray:
        """Convert patient feature dictionary to feature vector"""
        # This should match the feature engineering pipeline used in training
        # For now, return a simple vector - this would need to be adapted based on actual features

        feature_vector = [
            patient_features.get('n_patients', 50),
            patient_features.get('treatment_group', 25),
            patient_features.get('control_group', 25),
            patient_features.get('baseline_value', 0),
            patient_features.get('follow_up_months', 12),
            patient_features.get('safety_events', 0),
            # Add encoded categorical features
            patient_features.get('condition_encoded', 0),
            patient_features.get('intervention_encoded', 0),
            patient_features.get('phase_encoded', 1),
            patient_features.get('status_encoded', 0),
            patient_features.get('country_encoded', 0)
        ]

        return np.array(feature_vector)

    def _calculate_prediction_interval(self, model, feature_vector: np.ndarray,
                                     model_name: str, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate prediction interval for the prediction"""
        try:
            if hasattr(model, 'estimators_'):
                # For ensemble models, use variance across estimators
                predictions = [estimator.predict([feature_vector])[0]
                             for estimator in model.estimators_]
                pred_mean = np.mean(predictions)
                pred_std = np.std(predictions)

                # Calculate confidence interval
                alpha = 1 - confidence
                z_score = stats.norm.ppf(1 - alpha/2)

                lower = pred_mean - z_score * pred_std
                upper = pred_mean + z_score * pred_std

                return (lower, upper)

        except Exception:
            pass

        # Default: return None for non-ensemble models
        return (None, None)


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for treatment protocol optimization
    """

    def __init__(self, predictor: TreatmentOutcomePredictor, population_size: int = 50,
                 generations: int = 100, mutation_rate: float = 0.1):
        self.predictor = predictor
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        # Initialize DEAP framework
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

    def setup_ga(self, variable_params: List[str], param_ranges: Dict[str, Tuple[float, float]]):
        """Setup genetic algorithm components"""

        def create_individual():
            individual = []
            for param in variable_params:
                low, high = param_ranges[param]
                individual.append(random.uniform(low, high))
            return creator.Individual(individual)

        def evaluate_individual(individual):
            # Create patient profile with GA parameters
            patient_profile = {}
            for i, param in enumerate(variable_params):
                patient_profile[param] = individual[i]

            # Predict outcome
            prediction_result = self.predictor.predict_treatment_outcome(patient_profile)

            if 'error' in prediction_result:
                return (0.0,)  # Poor fitness for invalid parameters

            return (prediction_result['predicted_outcome'],)

        def mutate_individual(individual):
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    param = variable_params[i]
                    low, high = param_ranges[param]
                    individual[i] = random.uniform(low, high)
            return individual,

        # Register functions
        self.toolbox.register("individual", create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def optimize(self, variable_params: List[str], param_ranges: Dict[str, Tuple[float, float]]) -> Dict:
        """Run genetic algorithm optimization"""
        self.setup_ga(variable_params, param_ranges)

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution
        best_fitness_history = []

        for gen in range(self.generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:  # Crossover probability
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            # Track best fitness
            fits = [ind.fitness.values[0] for ind in population]
            best_fitness_history.append(max(fits))

        # Get best individual
        best_ind = tools.selBest(population, 1)[0]

        # Create optimized profile
        optimized_profile = {}
        for i, param in enumerate(variable_params):
            optimized_profile[param] = best_ind[i]

        return {
            'success': True,
            'optimized_parameters': optimized_profile,
            'best_fitness': best_ind.fitness.values[0],
            'fitness_history': best_fitness_history,
            'generations': self.generations
        }


class TreatmentOptimizer:
    """
    Optimize treatment protocols using predictive models
    """

    def __init__(self, predictor: TreatmentOutcomePredictor):
        self.predictor = predictor
        self.optimization_results = {}
        self.ga_optimizer = GeneticAlgorithmOptimizer(predictor)

    def optimize_treatment_protocol(self, base_patient: Dict,
                                  variable_params: List[str],
                                  param_ranges: Dict[str, Tuple[float, float]],
                                  objective: str = 'maximize_efficacy') -> Dict[str, Any]:
        """
        Optimize treatment parameters for a given patient profile
        """
        if not self.predictor.trained_models:
            return {'error': 'No trained models available for optimization'}

        def objective_function(params):
            # Create patient profile with optimized parameters
            patient_profile = base_patient.copy()

            for i, param_name in enumerate(variable_params):
                patient_profile[param_name] = params[i]

            # Predict outcome
            prediction_result = self.predictor.predict_treatment_outcome(patient_profile)

            if 'error' in prediction_result:
                return float('inf')  # Penalty for invalid parameters

            predicted_outcome = prediction_result['predicted_outcome']

            if objective == 'maximize_efficacy':
                return -predicted_outcome  # Minimize negative for maximization
            elif objective == 'minimize_risk':
                # Assume higher values are riskier (adapt based on your data)
                return predicted_outcome
            else:
                return -predicted_outcome

        # Set up optimization bounds
        bounds = [param_ranges[param] for param in variable_params]

        # Initial guess (midpoint of ranges)
        initial_guess = [(bounds[i][0] + bounds[i][1]) / 2 for i in range(len(bounds))]

        try:
            # Perform optimization
            result = minimize(
                objective_function,
                initial_guess,
                bounds=bounds,
                method='L-BFGS-B'
            )

            # Create optimized patient profile
            optimized_profile = base_patient.copy()
            for i, param_name in enumerate(variable_params):
                optimized_profile[param_name] = result.x[i]

            # Get prediction for optimized profile
            optimized_prediction = self.predictor.predict_treatment_outcome(optimized_profile)

            # Get baseline prediction
            baseline_prediction = self.predictor.predict_treatment_outcome(base_patient)

            optimization_results = {
                'success': result.success,
                'optimized_parameters': dict(zip(variable_params, result.x)),
                'optimized_profile': optimized_profile,
                'optimized_outcome': optimized_prediction.get('predicted_outcome'),
                'baseline_outcome': baseline_prediction.get('predicted_outcome'),
                'improvement': (optimized_prediction.get('predicted_outcome', 0) -
                              baseline_prediction.get('predicted_outcome', 0)),
                'optimization_details': {
                    'function_evaluations': result.nfev,
                    'final_objective_value': result.fun,
                    'convergence_message': result.message
                }
            }

            self.optimization_results = optimization_results
            return optimization_results

        except Exception as e:
            return {'error': f'Optimization failed: {str(e)}'}

    def genetic_algorithm_optimization(self, base_patient: Dict,
                                     variable_params: List[str],
                                     param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Optimize treatment parameters using genetic algorithm
        """
        if not self.predictor.trained_models:
            return {'error': 'No trained models available for optimization'}

        try:
            # Run genetic algorithm optimization
            ga_result = self.ga_optimizer.optimize(variable_params, param_ranges)

            if ga_result['success']:
                # Create optimized patient profile
                optimized_profile = base_patient.copy()
                optimized_profile.update(ga_result['optimized_parameters'])

                # Get predictions
                optimized_prediction = self.predictor.predict_treatment_outcome(optimized_profile)
                baseline_prediction = self.predictor.predict_treatment_outcome(base_patient)

                result = {
                    'success': True,
                    'method': 'genetic_algorithm',
                    'optimized_parameters': ga_result['optimized_parameters'],
                    'optimized_profile': optimized_profile,
                    'optimized_outcome': optimized_prediction.get('predicted_outcome'),
                    'baseline_outcome': baseline_prediction.get('predicted_outcome'),
                    'improvement': (optimized_prediction.get('predicted_outcome', 0) -
                                  baseline_prediction.get('predicted_outcome', 0)),
                    'best_fitness': ga_result['best_fitness'],
                    'fitness_history': ga_result['fitness_history'],
                    'generations': ga_result['generations']
                }

                return result

        except Exception as e:
            return {'error': f'Genetic algorithm optimization failed: {str(e)}'}

    def differential_evolution_optimization(self, base_patient: Dict,
                                          variable_params: List[str],
                                          param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Optimize treatment parameters using differential evolution
        """
        if not self.predictor.trained_models:
            return {'error': 'No trained models available for optimization'}

        def objective_function(params):
            # Create patient profile with optimized parameters
            patient_profile = base_patient.copy()

            for i, param_name in enumerate(variable_params):
                patient_profile[param_name] = params[i]

            # Predict outcome
            prediction_result = self.predictor.predict_treatment_outcome(patient_profile)

            if 'error' in prediction_result:
                return float('inf')

            predicted_outcome = prediction_result['predicted_outcome']
            return -predicted_outcome  # Minimize negative for maximization

        # Set up optimization bounds
        bounds = [param_ranges[param] for param in variable_params]

        try:
            # Perform differential evolution
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                atol=1e-6,
                seed=42
            )

            # Create optimized patient profile
            optimized_profile = base_patient.copy()
            for i, param_name in enumerate(variable_params):
                optimized_profile[param_name] = result.x[i]

            # Get predictions
            optimized_prediction = self.predictor.predict_treatment_outcome(optimized_profile)
            baseline_prediction = self.predictor.predict_treatment_outcome(base_patient)

            optimization_results = {
                'success': result.success,
                'method': 'differential_evolution',
                'optimized_parameters': dict(zip(variable_params, result.x)),
                'optimized_profile': optimized_profile,
                'optimized_outcome': optimized_prediction.get('predicted_outcome'),
                'baseline_outcome': baseline_prediction.get('predicted_outcome'),
                'improvement': (optimized_prediction.get('predicted_outcome', 0) -
                              baseline_prediction.get('predicted_outcome', 0)),
                'optimization_details': {
                    'function_evaluations': result.nfev,
                    'final_objective_value': result.fun,
                    'convergence_message': result.message
                }
            }

            return optimization_results

        except Exception as e:
            return {'error': f'Differential evolution optimization failed: {str(e)}'}

    def multi_objective_optimization(self, base_patient: Dict,
                                   variable_params: List[str],
                                   param_ranges: Dict[str, Tuple[float, float]],
                                   objectives: List[str] = ['efficacy', 'safety']) -> Dict[str, Any]:
        """
        Multi-objective optimization using NSGA-II algorithm
        """
        if not self.predictor.trained_models:
            return {'error': 'No trained models available for optimization'}

        # This is a simplified version - in practice, you'd need multiple models
        # for different objectives (efficacy, safety, cost, etc.)

        def evaluate_objectives(params):
            # Create patient profile with optimized parameters
            patient_profile = base_patient.copy()

            for i, param_name in enumerate(variable_params):
                patient_profile[param_name] = params[i]

            # Predict outcome (assuming this represents efficacy)
            prediction_result = self.predictor.predict_treatment_outcome(patient_profile)

            if 'error' in prediction_result:
                return [0.0, 0.0]  # Poor performance for invalid parameters

            efficacy = prediction_result['predicted_outcome']

            # Simplified safety calculation (in practice, use separate safety model)
            # Higher patient count might reduce safety (more complex to manage)
            safety = 100 - patient_profile.get('n_patients', 50) * 0.5

            return [efficacy, safety]

        # Run multiple single-objective optimizations to approximate Pareto front
        pareto_solutions = []

        # Weight combinations for scalarization
        weights = [
            [1.0, 0.0],    # Pure efficacy
            [0.8, 0.2],    # Efficacy-focused
            [0.6, 0.4],    # Balanced
            [0.4, 0.6],    # Safety-focused
            [0.2, 0.8],    # Safety-focused
            [0.0, 1.0]     # Pure safety
        ]

        for w_eff, w_safe in weights:
            def weighted_objective(params):
                efficacy, safety = evaluate_objectives(params)
                return -(w_eff * efficacy + w_safe * safety)

            bounds = [param_ranges[param] for param in variable_params]

            try:
                result = differential_evolution(
                    weighted_objective,
                    bounds,
                    maxiter=50,
                    popsize=10,
                    seed=42
                )

                if result.success:
                    efficacy, safety = evaluate_objectives(result.x)
                    optimized_params = dict(zip(variable_params, result.x))

                    pareto_solutions.append({
                        'parameters': optimized_params,
                        'efficacy': efficacy,
                        'safety': safety,
                        'weights': [w_eff, w_safe]
                    })

            except Exception:
                continue

        return {
            'success': len(pareto_solutions) > 0,
            'method': 'multi_objective',
            'pareto_solutions': pareto_solutions,
            'n_solutions': len(pareto_solutions)
        }


def main():
    """Main function to demonstrate predictive modeling"""
    print("Loading data for predictive modeling...")

    # Load data
    df = pd.read_csv('../data/clinical_trial_data.csv')

    # Initialize predictor
    predictor = TreatmentOutcomePredictor(target_variable='endpoint_value')

    print("\n=== TRAINING PREDICTIVE MODELS ===")
    training_results = predictor.train_all_models(df, optimize_hyperparams=True)

    if 'error' not in training_results:
        print(f"Successfully trained {len([k for k, v in training_results.items() if 'error' not in v])} models")

        # Display model rankings
        if 'model_ranking' in training_results:
            print("\nModel Performance Ranking:")
            for i, model_info in enumerate(training_results['model_ranking'][:5]):
                print(f"{i+1}. {model_info['model_name']}: "
                      f"R² = {model_info['test_r2']:.3f}, "
                      f"RMSE = {model_info['test_rmse']:.3f}")

        # Example prediction
        print("\n=== EXAMPLE TREATMENT PREDICTION ===")
        example_patient = {
            'n_patients': 50,
            'treatment_group': 25,
            'control_group': 25,
            'baseline_value': 10,
            'follow_up_months': 12,
            'safety_events': 1,
            'condition_encoded': 0,  # Epilepsy
            'intervention_encoded': 1,  # MSC
            'phase_encoded': 1,  # Phase 2
            'status_encoded': 1,  # Completed
            'country_encoded': 0   # USA
        }

        prediction_result = predictor.predict_treatment_outcome(example_patient)

        if 'error' not in prediction_result:
            print(f"Predicted outcome: {prediction_result['predicted_outcome']:.2f}")
            print(f"Model used: {prediction_result['model_used']}")

            if prediction_result['prediction_interval'][0] is not None:
                print(f"95% Prediction interval: "
                      f"[{prediction_result['prediction_interval'][0]:.2f}, "
                      f"{prediction_result['prediction_interval'][1]:.2f}]")

        # Treatment optimization example
        print("\n=== ADVANCED TREATMENT OPTIMIZATION ===")
        optimizer = TreatmentOptimizer(predictor)

        variable_params = ['follow_up_months', 'n_patients']
        param_ranges = {
            'follow_up_months': (6, 24),
            'n_patients': (10, 100)
        }

        # Standard optimization
        print("\n1. Standard L-BFGS-B Optimization:")
        optimization_result = optimizer.optimize_treatment_protocol(
            example_patient, variable_params, param_ranges
        )

        if 'error' not in optimization_result:
            print(f"Success: {optimization_result['success']}")
            print(f"Baseline outcome: {optimization_result['baseline_outcome']:.2f}")
            print(f"Optimized outcome: {optimization_result['optimized_outcome']:.2f}")
            print(f"Improvement: {optimization_result['improvement']:.2f}")

        # Genetic Algorithm optimization
        print("\n2. Genetic Algorithm Optimization:")
        ga_result = optimizer.genetic_algorithm_optimization(
            example_patient, variable_params, param_ranges
        )

        if 'error' not in ga_result and ga_result.get('success'):
            print(f"Best fitness: {ga_result['best_fitness']:.2f}")
            print(f"Generations: {ga_result['generations']}")
            print(f"Improvement: {ga_result['improvement']:.2f}")

        # Differential Evolution optimization
        print("\n3. Differential Evolution Optimization:")
        de_result = optimizer.differential_evolution_optimization(
            example_patient, variable_params, param_ranges
        )

        if 'error' not in de_result and de_result.get('success'):
            print(f"Function evaluations: {de_result['optimization_details']['function_evaluations']}")
            print(f"Improvement: {de_result['improvement']:.2f}")

        # Multi-objective optimization
        print("\n4. Multi-Objective Optimization:")
        mo_result = optimizer.multi_objective_optimization(
            example_patient, variable_params, param_ranges
        )

        if 'error' not in mo_result and mo_result.get('success'):
            print(f"Pareto solutions found: {mo_result['n_solutions']}")
            print("Sample solutions:")
            for i, solution in enumerate(mo_result['pareto_solutions'][:3]):
                print(f"  Solution {i+1}: Efficacy={solution['efficacy']:.2f}, "
                      f"Safety={solution['safety']:.2f}")

        # Model ensemble comparison
        print("\n=== ENSEMBLE MODEL PERFORMANCE ===")
        if 'model_ranking' in training_results:
            ensemble_models = [m for m in training_results['model_ranking']
                             if m['model_name'] in ['voting_regressor', 'bagging', 'stacking']]

            if ensemble_models:
                print("Ensemble model performance:")
                for model_info in ensemble_models:
                    print(f"{model_info['model_name']}: "
                          f"R² = {model_info['test_r2']:.3f}, "
                          f"RMSE = {model_info['test_rmse']:.3f}")

    else:
        print(f"Training failed: {training_results['error']}")

    print("\n" + "="*50)
    print("ADVANCED PREDICTIVE MODELING ANALYSIS COMPLETED!")
    print("="*50)


if __name__ == "__main__":
    main()