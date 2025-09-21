"""
Advanced Anomaly Detection System for Stem Cell Therapy Analysis

This module implements multiple anomaly detection algorithms to identify
unusual patterns, outliers, and behavioral anomalies in clinical trial data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore, chi2
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class MultiModalAnomalyDetector:
    """
    Comprehensive anomaly detection using multiple algorithms
    """

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detectors = {
            'isolation_forest': IsolationForest(contamination=contamination, random_state=42),
            'elliptic_envelope': EllipticEnvelope(contamination=contamination, random_state=42),
            'local_outlier_factor': LocalOutlierFactor(contamination=contamination),
            'dbscan': DBSCAN(eps=0.5, min_samples=2),
            'statistical_zscore': None  # Custom implementation
        }
        self.fitted_detectors = {}
        self.anomaly_scores = {}
        self.consensus_results = None

    def fit_detect(self, X: np.ndarray, method: str = 'all') -> Dict[str, np.ndarray]:
        """
        Fit detectors and detect anomalies

        Args:
            X: Feature matrix
            method: 'all' or specific detector name

        Returns:
            Dictionary of anomaly predictions (-1 for anomaly, 1 for normal)
        """
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = {}

        if method == 'all':
            methods_to_run = self.detectors.keys()
        else:
            methods_to_run = [method]

        for detector_name in methods_to_run:
            try:
                if detector_name == 'isolation_forest':
                    results[detector_name] = self._isolation_forest_detect(X_scaled)
                elif detector_name == 'elliptic_envelope':
                    results[detector_name] = self._elliptic_envelope_detect(X_scaled)
                elif detector_name == 'local_outlier_factor':
                    results[detector_name] = self._lof_detect(X_scaled)
                elif detector_name == 'dbscan':
                    results[detector_name] = self._dbscan_detect(X_scaled)
                elif detector_name == 'statistical_zscore':
                    results[detector_name] = self._zscore_detect(X_scaled)

            except Exception as e:
                print(f"Warning: {detector_name} failed with error: {e}")
                results[detector_name] = np.ones(len(X))  # Default to normal

        return results

    def _isolation_forest_detect(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest anomaly detection"""
        detector = self.detectors['isolation_forest']
        predictions = detector.fit_predict(X)
        self.fitted_detectors['isolation_forest'] = detector
        self.anomaly_scores['isolation_forest'] = detector.score_samples(X)
        return predictions

    def _elliptic_envelope_detect(self, X: np.ndarray) -> np.ndarray:
        """Elliptic Envelope anomaly detection"""
        if X.shape[1] < 2:
            return np.ones(len(X))

        detector = self.detectors['elliptic_envelope']
        predictions = detector.fit_predict(X)
        self.fitted_detectors['elliptic_envelope'] = detector
        return predictions

    def _lof_detect(self, X: np.ndarray) -> np.ndarray:
        """Local Outlier Factor detection"""
        detector = self.detectors['local_outlier_factor']
        predictions = detector.fit_predict(X)
        self.fitted_detectors['local_outlier_factor'] = detector
        return predictions

    def _dbscan_detect(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN clustering-based anomaly detection"""
        detector = self.detectors['dbscan']
        cluster_labels = detector.fit_predict(X)
        # Points labeled as -1 are noise/outliers
        predictions = np.where(cluster_labels == -1, -1, 1)
        self.fitted_detectors['dbscan'] = detector
        return predictions

    def _zscore_detect(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Statistical Z-score based detection"""
        z_scores = np.abs(zscore(X, axis=0))
        # Anomaly if any feature has |z-score| > threshold
        anomalies = np.any(z_scores > threshold, axis=1)
        predictions = np.where(anomalies, -1, 1)
        self.anomaly_scores['statistical_zscore'] = np.max(z_scores, axis=1)
        return predictions

    def consensus_detection(self, results: Dict[str, np.ndarray],
                          min_detectors: int = 2) -> np.ndarray:
        """
        Consensus anomaly detection across multiple methods

        Args:
            results: Dictionary of detection results
            min_detectors: Minimum number of detectors agreeing for consensus

        Returns:
            Consensus anomaly predictions
        """
        if not results:
            return np.array([])

        # Count how many detectors flag each point as anomaly
        anomaly_counts = np.zeros(len(list(results.values())[0]))

        for predictions in results.values():
            anomaly_counts += (predictions == -1).astype(int)

        # Consensus: anomaly if >= min_detectors agree
        consensus = np.where(anomaly_counts >= min_detectors, -1, 1)
        self.consensus_results = consensus

        return consensus


class CorrelationAnomalyDetector:
    """
    Detect anomalous correlations and relationships
    """

    def __init__(self):
        self.correlation_matrices = {}
        self.anomalous_pairs = []
        self.network_metrics = {}

    def detect_unusual_correlations(self, df: pd.DataFrame,
                                  threshold: float = 0.5,
                                  significance_level: float = 0.05) -> List[Dict]:
        """
        Detect statistically significant unusual correlations

        Args:
            df: Input dataframe
            threshold: Minimum correlation strength
            significance_level: P-value threshold

        Returns:
            List of unusual correlation findings
        """
        unusual_correlations = []

        # Calculate multiple correlation types
        methods = ['pearson', 'spearman', 'kendall']

        for method in methods:
            try:
                corr_matrix = df.corr(method=method)
                self.correlation_matrices[method] = corr_matrix

                # Find significant correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        var1 = corr_matrix.columns[i]
                        var2 = corr_matrix.columns[j]
                        correlation = corr_matrix.iloc[i, j]

                        if pd.isna(correlation) or abs(correlation) < threshold:
                            continue

                        # Calculate p-value
                        try:
                            if method == 'pearson':
                                _, p_value = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                            elif method == 'spearman':
                                _, p_value = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
                            elif method == 'kendall':
                                _, p_value = stats.kendalltau(df[var1].dropna(), df[var2].dropna())

                            if p_value < significance_level:
                                unusual_correlations.append({
                                    'method': method,
                                    'variable_1': var1,
                                    'variable_2': var2,
                                    'correlation': correlation,
                                    'p_value': p_value,
                                    'abs_correlation': abs(correlation),
                                    'strength': self._categorize_strength(abs(correlation)),
                                    'direction': 'positive' if correlation > 0 else 'negative'
                                })
                        except Exception as e:
                            continue

            except Exception as e:
                print(f"Failed to calculate {method} correlations: {e}")
                continue

        # Sort by absolute correlation strength
        unusual_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        self.anomalous_pairs = unusual_correlations

        return unusual_correlations

    def _categorize_strength(self, abs_corr: float) -> str:
        """Categorize correlation strength"""
        if abs_corr >= 0.8:
            return 'Very Strong'
        elif abs_corr >= 0.6:
            return 'Strong'
        elif abs_corr >= 0.4:
            return 'Moderate'
        else:
            return 'Weak'

    def build_correlation_network(self, min_correlation: float = 0.5) -> nx.Graph:
        """
        Build network graph of significant correlations

        Args:
            min_correlation: Minimum correlation strength for edge creation

        Returns:
            NetworkX graph object
        """
        G = nx.Graph()

        # Add edges for significant correlations
        for corr_info in self.anomalous_pairs:
            if corr_info['abs_correlation'] >= min_correlation:
                G.add_edge(
                    corr_info['variable_1'],
                    corr_info['variable_2'],
                    weight=corr_info['abs_correlation'],
                    correlation=corr_info['correlation'],
                    p_value=corr_info['p_value'],
                    method=corr_info['method']
                )

        # Calculate network metrics
        if len(G.nodes()) > 0:
            self.network_metrics = {
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G) if len(G.nodes()) > 1 else 0,
                'connected_components': nx.number_connected_components(G)
            }

            if len(G.nodes()) > 1:
                self.network_metrics['centrality'] = nx.degree_centrality(G)
                self.network_metrics['betweenness'] = nx.betweenness_centrality(G)

        return G


class TemporalAnomalyDetector:
    """
    Detect temporal anomalies and unusual time-based patterns
    """

    def __init__(self):
        self.temporal_patterns = {}
        self.anomalous_trends = []

    def detect_temporal_anomalies(self, df: pd.DataFrame,
                                 time_col: str = 'follow_up_months',
                                 value_col: str = 'endpoint_value') -> Dict[str, Any]:
        """
        Detect unusual temporal patterns

        Args:
            df: Input dataframe
            time_col: Column containing time values
            value_col: Column containing values to analyze over time

        Returns:
            Dictionary of temporal anomaly findings
        """
        temporal_data = df[[time_col, value_col]].dropna()

        if len(temporal_data) < 3:
            return {'error': 'Insufficient temporal data'}

        findings = {}

        # 1. Trend analysis
        correlation, p_value = stats.pearsonr(temporal_data[time_col], temporal_data[value_col])
        findings['trend_analysis'] = {
            'correlation': correlation,
            'p_value': p_value,
            'trend_direction': 'increasing' if correlation > 0 else 'decreasing',
            'significance': 'significant' if p_value < 0.05 else 'not_significant'
        }

        # 2. Detect temporal outliers using time-series decomposition
        temporal_outliers = self._detect_temporal_outliers(temporal_data, time_col, value_col)
        findings['temporal_outliers'] = temporal_outliers

        # 3. Seasonality/periodicity detection (if applicable)
        periodicity = self._detect_periodicity(temporal_data, time_col, value_col)
        findings['periodicity'] = periodicity

        # 4. Rate of change analysis
        rate_analysis = self._analyze_rate_of_change(temporal_data, time_col, value_col)
        findings['rate_analysis'] = rate_analysis

        self.temporal_patterns = findings
        return findings

    def _detect_temporal_outliers(self, df: pd.DataFrame, time_col: str, value_col: str) -> List[Dict]:
        """Detect outliers in temporal progression"""
        # Sort by time
        df_sorted = df.sort_values(time_col)

        # Calculate residuals from linear trend
        x = df_sorted[time_col].values
        y = df_sorted[value_col].values

        # Fit linear trend
        slope, intercept, _, _, _ = stats.linregress(x, y)
        predicted = slope * x + intercept
        residuals = y - predicted

        # Identify outliers (|residual| > 2*std)
        std_residual = np.std(residuals)
        outlier_threshold = 2 * std_residual

        outliers = []
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            if abs(residuals[i]) > outlier_threshold:
                outliers.append({
                    'index': idx,
                    'time_value': row[time_col],
                    'actual_value': row[value_col],
                    'predicted_value': predicted[i],
                    'residual': residuals[i],
                    'outlier_magnitude': abs(residuals[i]) / std_residual
                })

        return outliers

    def _detect_periodicity(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Detect periodic patterns in temporal data"""
        # Simple autocorrelation-based periodicity detection
        df_sorted = df.sort_values(time_col)
        values = df_sorted[value_col].values

        if len(values) < 4:
            return {'periodicity_detected': False, 'reason': 'insufficient_data'}

        # Calculate autocorrelation at different lags
        max_lag = min(len(values) // 2, 10)
        autocorrelations = []

        for lag in range(1, max_lag + 1):
            if len(values) > lag:
                corr, _ = stats.pearsonr(values[:-lag], values[lag:])
                autocorrelations.append({'lag': lag, 'correlation': corr})

        # Find significant autocorrelations
        significant_lags = [ac for ac in autocorrelations if abs(ac['correlation']) > 0.5]

        return {
            'periodicity_detected': len(significant_lags) > 0,
            'significant_lags': significant_lags,
            'all_autocorrelations': autocorrelations
        }

    def _analyze_rate_of_change(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Analyze rate of change patterns"""
        df_sorted = df.sort_values(time_col)

        if len(df_sorted) < 2:
            return {'error': 'insufficient_data_for_rate_analysis'}

        # Calculate differences
        time_diffs = np.diff(df_sorted[time_col].values)
        value_diffs = np.diff(df_sorted[value_col].values)

        # Calculate rates (avoiding division by zero)
        rates = np.where(time_diffs != 0, value_diffs / time_diffs, 0)

        # Detect unusual rate changes
        if len(rates) > 1:
            rate_mean = np.mean(rates)
            rate_std = np.std(rates)

            unusual_rates = []
            for i, rate in enumerate(rates):
                if abs(rate - rate_mean) > 2 * rate_std:
                    unusual_rates.append({
                        'interval_start': df_sorted.iloc[i][time_col],
                        'interval_end': df_sorted.iloc[i+1][time_col],
                        'rate': rate,
                        'z_score': (rate - rate_mean) / rate_std if rate_std > 0 else 0
                    })

            return {
                'mean_rate': rate_mean,
                'std_rate': rate_std,
                'unusual_rate_changes': unusual_rates,
                'total_intervals': len(rates)
            }

        return {'error': 'insufficient_intervals_for_analysis'}


class ComprehensiveAnomalyAnalyzer:
    """
    Main class that orchestrates all anomaly detection methods
    """

    def __init__(self, contamination: float = 0.1):
        self.multimodal_detector = MultiModalAnomalyDetector(contamination)
        self.correlation_detector = CorrelationAnomalyDetector()
        self.temporal_detector = TemporalAnomalyDetector()
        self.results = {}

    def analyze_dataframe(self, df: pd.DataFrame,
                         numerical_cols: List[str] = None,
                         categorical_cols: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive anomaly analysis of dataframe

        Args:
            df: Input dataframe
            numerical_cols: List of numerical columns to analyze
            categorical_cols: List of categorical columns to encode

        Returns:
            Comprehensive analysis results
        """
        print("Starting comprehensive anomaly analysis...")

        # Prepare data
        if numerical_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Encode categorical variables if provided
        analysis_df = df[numerical_cols].copy()

        if categorical_cols:
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    analysis_df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

        # Remove rows with all NaN values
        clean_df = analysis_df.dropna()

        if len(clean_df) < 3:
            return {'error': 'Insufficient clean data for analysis'}

        results = {}

        # 1. Multi-modal anomaly detection
        print("Performing multi-modal anomaly detection...")
        try:
            anomaly_results = self.multimodal_detector.fit_detect(clean_df.values)
            consensus = self.multimodal_detector.consensus_detection(anomaly_results, min_detectors=2)

            results['anomaly_detection'] = {
                'individual_methods': anomaly_results,
                'consensus': consensus,
                'anomaly_scores': self.multimodal_detector.anomaly_scores,
                'total_anomalies': (consensus == -1).sum(),
                'anomaly_rate': (consensus == -1).mean()
            }
        except Exception as e:
            print(f"Multi-modal detection failed: {e}")
            results['anomaly_detection'] = {'error': str(e)}

        # 2. Correlation anomaly detection
        print("Analyzing correlation anomalies...")
        try:
            unusual_correlations = self.correlation_detector.detect_unusual_correlations(clean_df)
            correlation_network = self.correlation_detector.build_correlation_network()

            results['correlation_analysis'] = {
                'unusual_correlations': unusual_correlations,
                'network_metrics': self.correlation_detector.network_metrics,
                'total_unusual_pairs': len(unusual_correlations)
            }
        except Exception as e:
            print(f"Correlation analysis failed: {e}")
            results['correlation_analysis'] = {'error': str(e)}

        # 3. Temporal anomaly detection
        print("Detecting temporal anomalies...")
        try:
            # Look for time-related columns
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'month' in col.lower() or 'date' in col.lower()]
            value_cols = [col for col in numerical_cols if 'endpoint' in col.lower() or 'value' in col.lower()]

            temporal_results = {}
            for time_col in time_cols[:2]:  # Analyze up to 2 time columns
                for value_col in value_cols[:2]:  # Analyze up to 2 value columns
                    if time_col in df.columns and value_col in df.columns:
                        key = f"{time_col}_vs_{value_col}"
                        temporal_results[key] = self.temporal_detector.detect_temporal_anomalies(df, time_col, value_col)

            results['temporal_analysis'] = temporal_results
        except Exception as e:
            print(f"Temporal analysis failed: {e}")
            results['temporal_analysis'] = {'error': str(e)}

        # 4. Statistical summary
        results['statistical_summary'] = {
            'dataset_shape': df.shape,
            'clean_data_shape': clean_df.shape,
            'data_completeness': len(clean_df) / len(df),
            'numerical_features': len(numerical_cols),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }

        self.results = results
        print("Comprehensive anomaly analysis completed!")

        return results

    def generate_anomaly_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive text report of anomaly findings

        Args:
            output_path: Optional path to save report

        Returns:
            Report text
        """
        if not self.results:
            return "No analysis results available. Run analyze_dataframe() first."

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE ANOMALY DETECTION REPORT")
        report_lines.append("="*80)
        report_lines.append("")

        # Statistical Summary
        if 'statistical_summary' in self.results:
            stats = self.results['statistical_summary']
            report_lines.append("DATASET OVERVIEW:")
            report_lines.append(f"  • Dataset shape: {stats['dataset_shape']}")
            report_lines.append(f"  • Clean data shape: {stats['clean_data_shape']}")
            report_lines.append(f"  • Data completeness: {stats['data_completeness']:.1%}")
            report_lines.append(f"  • Numerical features: {stats['numerical_features']}")
            report_lines.append("")

        # Anomaly Detection Results
        if 'anomaly_detection' in self.results and 'error' not in self.results['anomaly_detection']:
            anomaly = self.results['anomaly_detection']
            report_lines.append("ANOMALY DETECTION RESULTS:")
            report_lines.append(f"  • Total anomalies detected: {anomaly['total_anomalies']}")
            report_lines.append(f"  • Anomaly rate: {anomaly['anomaly_rate']:.1%}")
            report_lines.append("  • Detection methods:")

            for method, predictions in anomaly['individual_methods'].items():
                n_anomalies = (predictions == -1).sum()
                report_lines.append(f"    - {method}: {n_anomalies} anomalies")
            report_lines.append("")

        # Correlation Analysis
        if 'correlation_analysis' in self.results and 'error' not in self.results['correlation_analysis']:
            corr = self.results['correlation_analysis']
            report_lines.append("CORRELATION ANOMALIES:")
            report_lines.append(f"  • Unusual correlation pairs: {corr['total_unusual_pairs']}")

            if corr['unusual_correlations']:
                report_lines.append("  • Top 5 strongest correlations:")
                for i, pair in enumerate(corr['unusual_correlations'][:5]):
                    report_lines.append(f"    {i+1}. {pair['variable_1']} ↔ {pair['variable_2']}: "
                                      f"r={pair['correlation']:.3f} (p={pair['p_value']:.2e})")

            if 'network_metrics' in corr and corr['network_metrics']:
                metrics = corr['network_metrics']
                report_lines.append(f"  • Network density: {metrics.get('density', 0):.3f}")
                report_lines.append(f"  • Connected components: {metrics.get('connected_components', 0)}")
            report_lines.append("")

        # Temporal Analysis
        if 'temporal_analysis' in self.results and 'error' not in self.results['temporal_analysis']:
            temporal = self.results['temporal_analysis']
            report_lines.append("TEMPORAL ANOMALIES:")

            for analysis_key, analysis_result in temporal.items():
                if 'error' not in analysis_result:
                    report_lines.append(f"  • {analysis_key}:")

                    if 'trend_analysis' in analysis_result:
                        trend = analysis_result['trend_analysis']
                        report_lines.append(f"    - Trend: {trend['trend_direction']} "
                                          f"(r={trend['correlation']:.3f}, {trend['significance']})")

                    if 'temporal_outliers' in analysis_result:
                        outliers = analysis_result['temporal_outliers']
                        report_lines.append(f"    - Temporal outliers: {len(outliers)}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")

        # Anomaly investigation recommendations
        if 'anomaly_detection' in self.results and 'total_anomalies' in self.results['anomaly_detection']:
            n_anomalies = self.results['anomaly_detection']['total_anomalies']
            if n_anomalies > 0:
                report_lines.append(f"  • Investigate {n_anomalies} anomalous data points for potential data quality issues")

        # Correlation investigation recommendations
        if 'correlation_analysis' in self.results and 'total_unusual_pairs' in self.results['correlation_analysis']:
            n_correlations = self.results['correlation_analysis']['total_unusual_pairs']
            if n_correlations > 0:
                report_lines.append(f"  • Explore {n_correlations} unusual correlations for potential causal relationships")

        report_lines.append("  • Validate anomalous findings with domain experts")
        report_lines.append("  • Consider removing or treating anomalies for predictive modeling")
        report_lines.append("  • Use anomaly patterns for hypothesis generation")

        report_lines.append("")
        report_lines.append("="*80)

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")

        return report_text


def main():
    """Main function to demonstrate anomaly detection"""
    print("Loading clinical trial data for anomaly detection...")

    # Load data
    df = pd.read_csv('../data/clinical_trial_data.csv')

    # Define column types
    numerical_cols = ['n_patients', 'treatment_group', 'control_group', 'endpoint_value',
                     'baseline_value', 'percent_change', 'follow_up_months', 'safety_events']
    categorical_cols = ['condition', 'intervention', 'primary_endpoint', 'phase', 'status']

    # Initialize comprehensive analyzer
    analyzer = ComprehensiveAnomalyAnalyzer(contamination=0.15)

    # Perform analysis
    results = analyzer.analyze_dataframe(df, numerical_cols, categorical_cols)

    # Generate report
    report = analyzer.generate_anomaly_report('../results/anomaly_detection_report.txt')
    print("\n" + report)

    # Save results
    import json
    with open('../results/comprehensive_anomaly_results.json', 'w') as f:
        # Handle numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, default=convert_numpy, indent=2)

    print("Comprehensive anomaly analysis completed and saved!")


if __name__ == "__main__":
    main()