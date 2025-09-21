"""
Advanced Pattern Recognition System for Stem Cell Therapy Analysis

This module implements machine learning and AI algorithms to automatically
identify complex patterns, trends, and anomalies in clinical trial data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class TemporalPatternRecognizer:
    """
    Recognize temporal patterns and trends in clinical trial data
    """

    def __init__(self):
        self.patterns = {}
        self.trend_models = {}

    def detect_temporal_patterns(self, df: pd.DataFrame,
                                time_col: str = 'follow_up_months',
                                value_col: str = 'endpoint_value') -> Dict[str, Any]:
        """
        Detect complex temporal patterns using multiple algorithms
        """
        temporal_data = df[[time_col, value_col]].dropna().sort_values(time_col)

        if len(temporal_data) < 4:
            return {'error': 'Insufficient temporal data'}

        patterns = {}

        # 1. Trend Analysis with Change Point Detection
        patterns['trend_analysis'] = self._detect_trend_changes(temporal_data, time_col, value_col)

        # 2. Seasonality and Periodicity
        patterns['periodicity'] = self._detect_periodicity_advanced(temporal_data, time_col, value_col)

        # 3. Volatility Clustering
        patterns['volatility'] = self._detect_volatility_clusters(temporal_data, time_col, value_col)

        # 4. Regime Switching Detection
        patterns['regime_switching'] = self._detect_regime_switches(temporal_data, time_col, value_col)

        # 5. Non-linear Trend Detection
        patterns['nonlinear_trends'] = self._detect_nonlinear_trends(temporal_data, time_col, value_col)

        self.patterns = patterns
        return patterns

    def _detect_trend_changes(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Detect change points in temporal trends"""
        from scipy.signal import find_peaks

        # Calculate rolling derivatives to find trend changes
        window_size = max(3, len(df) // 4)
        rolling_slope = []
        time_points = []

        for i in range(window_size, len(df)):
            window_data = df.iloc[i-window_size:i]
            if len(window_data) > 1:
                slope, _, _, _, _ = stats.linregress(window_data[time_col], window_data[value_col])
                rolling_slope.append(slope)
                time_points.append(window_data[time_col].iloc[-1])

        if len(rolling_slope) < 3:
            return {'change_points': [], 'trend_segments': []}

        # Find peaks in absolute slope changes
        slope_changes = np.abs(np.diff(rolling_slope))
        peaks, _ = find_peaks(slope_changes, height=np.std(slope_changes))

        change_points = []
        for peak in peaks:
            if peak < len(time_points):
                change_points.append({
                    'time': time_points[peak],
                    'slope_before': rolling_slope[peak],
                    'slope_after': rolling_slope[peak+1] if peak+1 < len(rolling_slope) else None,
                    'magnitude': slope_changes[peak]
                })

        return {
            'change_points': change_points,
            'rolling_slopes': rolling_slope,
            'time_points': time_points,
            'trend_segments': len(change_points) + 1
        }

    def _detect_periodicity_advanced(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Advanced periodicity detection using spectral analysis"""
        values = df[value_col].values
        times = df[time_col].values

        if len(values) < 6:
            return {'periodic': False, 'reason': 'insufficient_data'}

        # Interpolate to regular time grid for FFT
        from scipy.interpolate import interp1d
        time_regular = np.linspace(times.min(), times.max(), len(times))
        f_interp = interp1d(times, values, kind='linear', fill_value='extrapolate')
        values_regular = f_interp(time_regular)

        # FFT analysis
        fft_values = np.fft.fft(values_regular)
        frequencies = np.fft.fftfreq(len(values_regular))

        # Find dominant frequencies
        power_spectrum = np.abs(fft_values)**2
        dominant_freq_idx = np.argsort(power_spectrum[1:len(power_spectrum)//2])[-3:] + 1

        periods = []
        for idx in dominant_freq_idx:
            if frequencies[idx] != 0:
                period = 1 / abs(frequencies[idx])
                periods.append({
                    'period': period * (times.max() - times.min()) / len(times),
                    'strength': power_spectrum[idx] / np.sum(power_spectrum),
                    'frequency': frequencies[idx]
                })

        # Statistical significance test
        significant_periods = [p for p in periods if p['strength'] > 0.1]

        return {
            'periodic': len(significant_periods) > 0,
            'periods': significant_periods,
            'dominant_period': significant_periods[0] if significant_periods else None,
            'spectral_entropy': -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        }

    def _detect_volatility_clusters(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Detect clustering in volatility patterns"""
        if len(df) < 5:
            return {'volatility_clusters': False}

        # Calculate rolling volatility (standard deviation)
        window_size = max(3, len(df) // 3)
        volatilities = []
        time_points = []

        for i in range(window_size, len(df)):
            window_data = df.iloc[i-window_size:i]
            vol = window_data[value_col].std()
            volatilities.append(vol)
            time_points.append(window_data[time_col].iloc[-1])

        if len(volatilities) < 3:
            return {'volatility_clusters': False}

        # Cluster volatility values
        from sklearn.cluster import KMeans
        volatilities_array = np.array(volatilities).reshape(-1, 1)

        # Optimal number of clusters
        silhouette_scores = []
        K_range = range(2, min(6, len(volatilities)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(volatilities_array)
            score = silhouette_score(volatilities_array, cluster_labels)
            silhouette_scores.append(score)

        if silhouette_scores:
            best_k = K_range[np.argmax(silhouette_scores)]
            kmeans_best = KMeans(n_clusters=best_k, random_state=42)
            clusters = kmeans_best.fit_predict(volatilities_array)

            cluster_info = []
            for i in range(best_k):
                cluster_mask = clusters == i
                cluster_info.append({
                    'cluster_id': i,
                    'mean_volatility': np.mean(np.array(volatilities)[cluster_mask]),
                    'time_periods': np.array(time_points)[cluster_mask].tolist(),
                    'size': np.sum(cluster_mask)
                })

            return {
                'volatility_clusters': True,
                'n_clusters': best_k,
                'cluster_info': cluster_info,
                'silhouette_score': max(silhouette_scores)
            }

        return {'volatility_clusters': False}

    def _detect_regime_switches(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Detect regime switching using statistical methods"""
        values = df[value_col].values

        if len(values) < 6:
            return {'regime_switches': False}

        # Simple regime detection using rolling statistics
        window_size = max(3, len(values) // 4)
        regimes = []

        for i in range(window_size, len(values)):
            window = values[i-window_size:i]

            regime_stats = {
                'start_time': df[time_col].iloc[i-window_size],
                'end_time': df[time_col].iloc[i-1],
                'mean': np.mean(window),
                'std': np.std(window),
                'trend': stats.linregress(range(len(window)), window)[0]
            }
            regimes.append(regime_stats)

        # Detect regime changes based on statistical differences
        regime_changes = []
        for i in range(1, len(regimes)):
            prev_regime = regimes[i-1]
            curr_regime = regimes[i]

            # Test for significant change in mean
            mean_change = abs(curr_regime['mean'] - prev_regime['mean'])
            pooled_std = np.sqrt((prev_regime['std']**2 + curr_regime['std']**2) / 2)

            if pooled_std > 0:
                z_score = mean_change / pooled_std
                if z_score > 1.96:  # Significant at 95% level
                    regime_changes.append({
                        'change_time': curr_regime['start_time'],
                        'mean_before': prev_regime['mean'],
                        'mean_after': curr_regime['mean'],
                        'significance': z_score
                    })

        return {
            'regime_switches': len(regime_changes) > 0,
            'n_switches': len(regime_changes),
            'switches': regime_changes,
            'regimes': regimes
        }

    def _detect_nonlinear_trends(self, df: pd.DataFrame, time_col: str, value_col: str) -> Dict:
        """Detect non-linear trends using polynomial fitting"""
        if len(df) < 4:
            return {'nonlinear': False}

        x = df[time_col].values
        y = df[value_col].values

        # Fit polynomials of different degrees
        degrees = [1, 2, 3]
        models = {}

        for degree in degrees:
            if len(x) > degree:
                coeffs = np.polyfit(x, y, degree)
                poly = np.poly1d(coeffs)

                # Calculate goodness of fit
                y_pred = poly(x)
                r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

                models[f'degree_{degree}'] = {
                    'coefficients': coeffs.tolist(),
                    'r_squared': r2,
                    'aic': len(x) * np.log(np.sum((y - y_pred)**2) / len(x)) + 2 * (degree + 1)
                }

        # Select best model based on AIC
        if models:
            best_model = min(models.keys(), key=lambda k: models[k]['aic'])
            best_degree = int(best_model.split('_')[1])

            return {
                'nonlinear': best_degree > 1,
                'best_degree': best_degree,
                'models': models,
                'improvement_over_linear': models[best_model]['r_squared'] - models['degree_1']['r_squared'] if 'degree_1' in models else 0
            }

        return {'nonlinear': False}


class ClusterPatternAnalyzer:
    """
    Advanced clustering analysis to identify hidden patterns in trial data
    """

    def __init__(self):
        self.clustering_results = {}
        self.optimal_clusters = {}

    def comprehensive_clustering_analysis(self, df: pd.DataFrame,
                                        feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive clustering analysis using multiple algorithms
        """
        # Prepare data
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        analysis_df = df[feature_cols].dropna()

        if len(analysis_df) < 3:
            return {'error': 'Insufficient data for clustering'}

        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)

        results = {}

        # 1. K-Means clustering with optimal K detection
        results['kmeans'] = self._kmeans_analysis(scaled_data, analysis_df.index)

        # 2. Hierarchical clustering
        results['hierarchical'] = self._hierarchical_analysis(scaled_data, analysis_df.index)

        # 3. DBSCAN for density-based clustering
        results['dbscan'] = self._dbscan_analysis(scaled_data, analysis_df.index)

        # 4. Gaussian Mixture Model
        results['gaussian_mixture'] = self._gaussian_mixture_analysis(scaled_data, analysis_df.index)

        # 5. Consensus clustering
        results['consensus'] = self._consensus_clustering(results)

        # 6. Feature importance for clustering
        results['feature_importance'] = self._clustering_feature_importance(scaled_data, feature_cols)

        self.clustering_results = results
        return results

    def _kmeans_analysis(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """K-means clustering with optimal K selection"""
        max_k = min(10, len(X) // 2)
        if max_k < 2:
            return {'error': 'Insufficient samples for K-means'}

        # Elbow method and silhouette analysis
        inertias = []
        silhouette_scores = []
        calinski_scores = []

        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)

            if len(set(cluster_labels)) > 1:  # Ensure multiple clusters
                sil_score = silhouette_score(X, cluster_labels)
                cal_score = calinski_harabasz_score(X, cluster_labels)
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
            else:
                silhouette_scores.append(-1)
                calinski_scores.append(0)

        # Find optimal K
        if silhouette_scores:
            optimal_k_sil = K_range[np.argmax(silhouette_scores)]
            optimal_k_cal = K_range[np.argmax(calinski_scores)]

            # Use silhouette score as primary criterion
            optimal_k = optimal_k_sil

            # Fit final model
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(X)

            return {
                'optimal_k': optimal_k,
                'cluster_labels': dict(zip(sample_indices, final_labels)),
                'cluster_centers': final_kmeans.cluster_centers_,
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'best_silhouette': max(silhouette_scores),
                'final_inertia': final_kmeans.inertia_
            }

        return {'error': 'Clustering evaluation failed'}

    def _hierarchical_analysis(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """Hierarchical clustering analysis"""
        if len(X) < 3:
            return {'error': 'Insufficient data for hierarchical clustering'}

        # Calculate linkage matrix
        linkage_matrix = linkage(X, method='ward')

        # Determine optimal number of clusters using inconsistency
        max_clusters = min(8, len(X) // 2)
        cluster_range = range(2, max_clusters + 1)

        best_score = -1
        best_n_clusters = 2

        for n_clusters in cluster_range:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

            if len(set(cluster_labels)) > 1:
                score = silhouette_score(X, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters

        # Get final clustering
        final_labels = fcluster(linkage_matrix, best_n_clusters, criterion='maxclust')

        return {
            'optimal_clusters': best_n_clusters,
            'cluster_labels': dict(zip(sample_indices, final_labels)),
            'linkage_matrix': linkage_matrix,
            'silhouette_score': best_score
        }

    def _dbscan_analysis(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """DBSCAN density-based clustering"""
        if len(X) < 3:
            return {'error': 'Insufficient data for DBSCAN'}

        # Parameter optimization for DBSCAN
        from sklearn.neighbors import NearestNeighbors

        # Find optimal eps using k-distance graph
        k = min(4, len(X) - 1)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)

        # Sort distances to k-th neighbor
        k_distances = distances[:, -1]
        k_distances = np.sort(k_distances)

        # Find knee point (simplified)
        eps_optimal = np.percentile(k_distances, 75)

        # Try different min_samples values
        min_samples_range = range(2, min(6, len(X) // 2))
        best_score = -1
        best_params = None

        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X)

            # Check if we have meaningful clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            if n_clusters > 1 and n_clusters < len(X) // 2:
                # Calculate silhouette score (excluding noise points)
                mask = cluster_labels != -1
                if np.sum(mask) > 1:
                    score = silhouette_score(X[mask], cluster_labels[mask])
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps_optimal, 'min_samples': min_samples}

        if best_params:
            final_dbscan = DBSCAN(**best_params)
            final_labels = final_dbscan.fit_predict(X)

            n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
            n_noise = list(final_labels).count(-1)

            return {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_labels': dict(zip(sample_indices, final_labels)),
                'parameters': best_params,
                'silhouette_score': best_score
            }

        return {'error': 'No suitable DBSCAN parameters found'}

    def _gaussian_mixture_analysis(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """Gaussian Mixture Model clustering"""
        from sklearn.mixture import GaussianMixture

        max_components = min(8, len(X) // 2)
        if max_components < 2:
            return {'error': 'Insufficient data for GMM'}

        # Model selection using information criteria
        n_components_range = range(2, max_components + 1)
        aic_scores = []
        bic_scores = []

        best_aic = np.inf
        best_model = None

        for n_components in n_components_range:
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(X)

                aic = gmm.aic(X)
                bic = gmm.bic(X)

                aic_scores.append(aic)
                bic_scores.append(bic)

                if aic < best_aic:
                    best_aic = aic
                    best_model = gmm

            except Exception:
                aic_scores.append(np.inf)
                bic_scores.append(np.inf)

        if best_model:
            cluster_labels = best_model.predict(X)
            cluster_probs = best_model.predict_proba(X)

            return {
                'optimal_components': best_model.n_components,
                'cluster_labels': dict(zip(sample_indices, cluster_labels)),
                'cluster_probabilities': cluster_probs,
                'aic_scores': aic_scores,
                'bic_scores': bic_scores,
                'best_aic': best_aic,
                'converged': best_model.converged_
            }

        return {'error': 'GMM fitting failed'}

    def _consensus_clustering(self, clustering_results: Dict) -> Dict:
        """Create consensus clustering from multiple methods"""
        # Collect all clustering results
        all_labelings = {}

        for method, results in clustering_results.items():
            if 'cluster_labels' in results:
                all_labelings[method] = results['cluster_labels']

        if len(all_labelings) < 2:
            return {'error': 'Insufficient clustering results for consensus'}

        # Find common samples
        common_indices = set.intersection(*[set(labels.keys()) for labels in all_labelings.values()])

        if len(common_indices) < 3:
            return {'error': 'Insufficient common samples for consensus'}

        # Create co-association matrix
        n_samples = len(common_indices)
        indices_list = list(common_indices)
        coassoc_matrix = np.zeros((n_samples, n_samples))

        for method_labels in all_labelings.values():
            for i, idx1 in enumerate(indices_list):
                for j, idx2 in enumerate(indices_list):
                    if method_labels[idx1] == method_labels[idx2]:
                        coassoc_matrix[i, j] += 1

        # Normalize by number of methods
        coassoc_matrix /= len(all_labelings)

        # Apply threshold and cluster
        threshold = 0.5  # Majority vote
        consensus_adjacency = coassoc_matrix > threshold

        # Use connected components for final clustering
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        n_components, consensus_labels = connected_components(
            sp.csr_matrix(consensus_adjacency)
        )

        consensus_dict = dict(zip(indices_list, consensus_labels))

        return {
            'n_consensus_clusters': n_components,
            'consensus_labels': consensus_dict,
            'coassociation_matrix': coassoc_matrix,
            'agreement_level': np.mean(coassoc_matrix > threshold)
        }

    def _clustering_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Determine feature importance for clustering"""
        from sklearn.ensemble import RandomForestClassifier

        # Use the best clustering result as labels
        if 'kmeans' in self.clustering_results and 'cluster_labels' in self.clustering_results['kmeans']:
            labels = list(self.clustering_results['kmeans']['cluster_labels'].values())

            if len(set(labels)) > 1:
                # Train RF to predict cluster membership
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, labels)

                feature_importance = dict(zip(feature_names, rf.feature_importances_))

                return {
                    'feature_importance': feature_importance,
                    'most_important': max(feature_importance.keys(), key=feature_importance.get),
                    'least_important': min(feature_importance.keys(), key=feature_importance.get),
                    'importance_scores': rf.feature_importances_
                }

        return {'error': 'Cannot compute feature importance'}


class AnomalyPatternDetector:
    """
    Advanced anomaly pattern detection using ensemble methods
    """

    def __init__(self):
        self.detectors = {}
        self.anomaly_patterns = {}

    def detect_complex_anomalies(self, df: pd.DataFrame,
                                feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Detect complex anomaly patterns using multiple approaches
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        analysis_df = df[feature_cols].dropna()

        if len(analysis_df) < 5:
            return {'error': 'Insufficient data for anomaly detection'}

        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(analysis_df)

        results = {}

        # 1. Statistical anomalies
        results['statistical'] = self._statistical_anomalies(scaled_data, analysis_df.index)

        # 2. Distance-based anomalies
        results['distance_based'] = self._distance_based_anomalies(scaled_data, analysis_df.index)

        # 3. Density-based anomalies
        results['density_based'] = self._density_based_anomalies(scaled_data, analysis_df.index)

        # 4. Isolation-based anomalies
        results['isolation_based'] = self._isolation_based_anomalies(scaled_data, analysis_df.index)

        # 5. Ensemble anomaly detection
        results['ensemble'] = self._ensemble_anomaly_detection(results)

        # 6. Anomaly characterization
        results['characterization'] = self._characterize_anomalies(
            scaled_data, analysis_df.index, feature_cols, results['ensemble']
        )

        self.anomaly_patterns = results
        return results

    def _statistical_anomalies(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """Detect statistical anomalies using Z-score and modified Z-score"""
        # Standard Z-score
        z_scores = np.abs(stats.zscore(X, axis=0))
        z_anomalies = np.any(z_scores > 3, axis=1)

        # Modified Z-score (more robust)
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        modified_z_scores = 0.6745 * (X - median) / mad
        modified_z_anomalies = np.any(np.abs(modified_z_scores) > 3.5, axis=1)

        # Mahalanobis distance
        try:
            cov_matrix = np.cov(X.T)
            inv_cov = np.linalg.pinv(cov_matrix)
            mean = np.mean(X, axis=0)

            mahal_distances = []
            for x in X:
                diff = x - mean
                mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                mahal_distances.append(mahal_dist)

            mahal_threshold = np.percentile(mahal_distances, 95)
            mahal_anomalies = np.array(mahal_distances) > mahal_threshold

        except Exception:
            mahal_anomalies = np.zeros(len(X), dtype=bool)
            mahal_distances = np.zeros(len(X))

        return {
            'z_score_anomalies': dict(zip(sample_indices, z_anomalies)),
            'modified_z_anomalies': dict(zip(sample_indices, modified_z_anomalies)),
            'mahalanobis_anomalies': dict(zip(sample_indices, mahal_anomalies)),
            'z_scores': z_scores,
            'mahalanobis_distances': mahal_distances
        }

    def _distance_based_anomalies(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """Detect anomalies based on distance to nearest neighbors"""
        from sklearn.neighbors import NearestNeighbors

        k = min(5, len(X) - 1)
        if k < 1:
            return {'error': 'Insufficient data for distance-based detection'}

        nbrs = NearestNeighbors(n_neighbors=k + 1)  # +1 because point is its own neighbor
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Use distance to k-th nearest neighbor
        kth_distances = distances[:, -1]

        # Anomalies are points with distances in top percentile
        threshold = np.percentile(kth_distances, 90)
        distance_anomalies = kth_distances > threshold

        return {
            'distance_anomalies': dict(zip(sample_indices, distance_anomalies)),
            'kth_distances': kth_distances,
            'threshold': threshold
        }

    def _density_based_anomalies(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """Detect anomalies using Local Outlier Factor"""
        from sklearn.neighbors import LocalOutlierFactor

        if len(X) < 3:
            return {'error': 'Insufficient data for density-based detection'}

        n_neighbors = min(5, len(X) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)

        try:
            outlier_labels = lof.fit_predict(X)
            outlier_scores = -lof.negative_outlier_factor_

            lof_anomalies = outlier_labels == -1

            return {
                'lof_anomalies': dict(zip(sample_indices, lof_anomalies)),
                'lof_scores': outlier_scores,
                'n_neighbors': n_neighbors
            }
        except Exception as e:
            return {'error': f'LOF detection failed: {str(e)}'}

    def _isolation_based_anomalies(self, X: np.ndarray, sample_indices: pd.Index) -> Dict:
        """Detect anomalies using Isolation Forest"""
        if len(X) < 3:
            return {'error': 'Insufficient data for isolation-based detection'}

        iso_forest = IsolationForest(contamination=0.1, random_state=42)

        try:
            outlier_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.decision_function(X)

            isolation_anomalies = outlier_labels == -1

            return {
                'isolation_anomalies': dict(zip(sample_indices, isolation_anomalies)),
                'anomaly_scores': anomaly_scores,
                'contamination': 0.1
            }
        except Exception as e:
            return {'error': f'Isolation Forest detection failed: {str(e)}'}

    def _ensemble_anomaly_detection(self, individual_results: Dict) -> Dict:
        """Combine multiple anomaly detection methods"""
        # Collect all anomaly indicators
        anomaly_methods = {}

        for method, results in individual_results.items():
            if isinstance(results, dict):
                for submethod, anomalies in results.items():
                    if 'anomalies' in submethod and isinstance(anomalies, dict):
                        anomaly_methods[f'{method}_{submethod}'] = anomalies

        if not anomaly_methods:
            return {'error': 'No anomaly detection results to ensemble'}

        # Find common indices
        all_indices = set.intersection(*[set(anomalies.keys()) for anomalies in anomaly_methods.values()])

        if not all_indices:
            return {'error': 'No common indices across methods'}

        # Count votes for each sample
        anomaly_votes = {}
        for idx in all_indices:
            votes = sum([1 for anomalies in anomaly_methods.values() if anomalies.get(idx, False)])
            anomaly_votes[idx] = votes

        # Determine consensus anomalies (majority vote)
        n_methods = len(anomaly_methods)
        consensus_threshold = n_methods // 2 + 1

        consensus_anomalies = {idx: votes >= consensus_threshold
                             for idx, votes in anomaly_votes.items()}

        return {
            'consensus_anomalies': consensus_anomalies,
            'anomaly_votes': anomaly_votes,
            'n_methods': n_methods,
            'consensus_threshold': consensus_threshold,
            'method_agreement': np.mean([votes / n_methods for votes in anomaly_votes.values()])
        }

    def _characterize_anomalies(self, X: np.ndarray, sample_indices: pd.Index,
                               feature_names: List[str], ensemble_results: Dict) -> Dict:
        """Characterize detected anomalies"""
        if 'consensus_anomalies' not in ensemble_results:
            return {'error': 'No consensus anomalies to characterize'}

        consensus_anomalies = ensemble_results['consensus_anomalies']

        # Find anomalous samples
        anomaly_indices = [idx for idx, is_anomaly in consensus_anomalies.items() if is_anomaly]

        if not anomaly_indices:
            return {'no_anomalies': True}

        # Get positions in original array
        index_to_pos = {idx: pos for pos, idx in enumerate(sample_indices)}
        anomaly_positions = [index_to_pos[idx] for idx in anomaly_indices if idx in index_to_pos]

        if not anomaly_positions:
            return {'error': 'Cannot map anomaly indices to data positions'}

        # Characterize anomalies
        normal_data = np.delete(X, anomaly_positions, axis=0)
        anomaly_data = X[anomaly_positions]

        characterization = {}

        for i, feature in enumerate(feature_names):
            normal_values = normal_data[:, i] if len(normal_data) > 0 else []
            anomaly_values = anomaly_data[:, i]

            if len(normal_values) > 0:
                feature_char = {
                    'normal_mean': np.mean(normal_values),
                    'normal_std': np.std(normal_values),
                    'anomaly_mean': np.mean(anomaly_values),
                    'anomaly_std': np.std(anomaly_values),
                    'difference': np.mean(anomaly_values) - np.mean(normal_values)
                }

                # Statistical test for difference
                if len(normal_values) > 1 and len(anomaly_values) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(anomaly_values, normal_values)
                        feature_char['t_statistic'] = t_stat
                        feature_char['p_value'] = p_value
                    except:
                        feature_char['t_statistic'] = None
                        feature_char['p_value'] = None

                characterization[feature] = feature_char

        return {
            'n_anomalies': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'feature_characterization': characterization
        }


def main():
    """Main function to demonstrate pattern recognition"""
    print("Loading data for advanced pattern recognition...")

    # Load data
    df = pd.read_csv('../data/clinical_trial_data.csv')

    # Initialize pattern recognizers
    temporal_recognizer = TemporalPatternRecognizer()
    cluster_analyzer = ClusterPatternAnalyzer()
    anomaly_detector = AnomalyPatternDetector()

    print("\n=== TEMPORAL PATTERN RECOGNITION ===")
    temporal_patterns = temporal_recognizer.detect_temporal_patterns(df)

    if 'error' not in temporal_patterns:
        for pattern_type, results in temporal_patterns.items():
            print(f"\n{pattern_type.upper()}:")
            if isinstance(results, dict) and 'error' not in results:
                for key, value in results.items():
                    if isinstance(value, (int, float, str, bool)):
                        print(f"  {key}: {value}")

    print("\n=== CLUSTERING ANALYSIS ===")
    clustering_results = cluster_analyzer.comprehensive_clustering_analysis(df)

    if 'error' not in clustering_results:
        for method, results in clustering_results.items():
            if isinstance(results, dict) and 'error' not in results:
                print(f"\n{method.upper()}: {len(results)} metrics available")

    print("\n=== ANOMALY PATTERN DETECTION ===")
    anomaly_results = anomaly_detector.detect_complex_anomalies(df)

    if 'error' not in anomaly_results:
        if 'ensemble' in anomaly_results and 'consensus_anomalies' in anomaly_results['ensemble']:
            n_anomalies = sum(anomaly_results['ensemble']['consensus_anomalies'].values())
            print(f"Consensus anomalies detected: {n_anomalies}")

    # Save results
    import json

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results = {
        'temporal_patterns': temporal_patterns,
        'clustering_results': clustering_results,
        'anomaly_results': anomaly_results
    }

    with open('../results/pattern_recognition_results.json', 'w') as f:
        json.dump(all_results, f, default=convert_numpy, indent=2)

    print("\nPattern recognition analysis completed and saved!")


if __name__ == "__main__":
    main()