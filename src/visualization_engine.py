"""
Advanced Visualization Engine for Stem Cell Therapy Analysis

This module creates comprehensive interactive visualizations, charts, and graphs
to reveal hidden patterns, correlations, and anomalies in clinical trial data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class StatisticalVisualizationEngine:
    """
    Comprehensive visualization engine for statistical analysis
    """

    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        plt.style.use('default')  # Reset to default first
        self.figsize = figsize
        self.color_palette = sns.color_palette("Set2", 12)
        sns.set_palette(self.color_palette)

        # Configuration
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10

    def create_correlation_heatmap_suite(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        Create comprehensive correlation heatmap visualizations
        """
        print("Creating correlation heatmap suite...")

        # Remove non-numeric columns and clean data
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if len(numeric_df.columns) < 2:
            print("Insufficient numeric columns for correlation analysis")
            return

        # Calculate different correlation types
        correlations = {
            'Pearson': numeric_df.corr(method='pearson'),
            'Spearman': numeric_df.corr(method='spearman'),
            'Kendall': numeric_df.corr(method='kendall')
        }

        # Create multi-panel correlation plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Individual correlation matrices
        for i, (method, corr_matrix) in enumerate(list(correlations.items())[:3]):
            row, col = (i // 2, i % 2) if i < 2 else (1, 0 if i == 2 else 1)

            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix,
                       mask=mask,
                       annot=True,
                       cmap='RdYlBu_r',
                       center=0,
                       fmt='.2f',
                       square=True,
                       ax=axes[row, col],
                       cbar_kws={'shrink': 0.8})
            axes[row, col].set_title(f'{method} Correlation Matrix', fontweight='bold', fontsize=14)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].tick_params(axis='y', rotation=0)

        # Correlation strength distribution
        all_correlations = []
        for method, corr_matrix in correlations.items():
            # Extract upper triangle correlations (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_values = corr_matrix.values[mask]
            corr_values = corr_values[~np.isnan(corr_values)]
            all_correlations.extend([(method, val) for val in corr_values])

        if all_correlations:
            corr_df = pd.DataFrame(all_correlations, columns=['Method', 'Correlation'])

            # Distribution plot
            sns.boxplot(data=corr_df, x='Method', y='Correlation', ax=axes[1, 1])
            axes[1, 1].set_title('Correlation Strength Distribution by Method', fontweight='bold')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/correlation_heatmap_suite.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_anomaly_detection_plots(self, df: pd.DataFrame, anomaly_results: dict,
                                     save_path: str = None) -> None:
        """
        Create comprehensive anomaly detection visualizations
        """
        print("Creating anomaly detection visualizations...")

        if not anomaly_results or 'individual_methods' not in anomaly_results:
            print("No anomaly results available for visualization")
            return

        # Prepare data for plotting
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if len(numeric_df) < 3:
            print("Insufficient data for anomaly visualization")
            return

        # Perform PCA for 2D visualization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=min(2, scaled_data.shape[1]))
        pca_data = pca.fit_transform(scaled_data)

        # Create multi-panel anomaly visualization
        methods = list(anomaly_results['individual_methods'].keys())
        n_methods = len(methods)

        # Calculate grid dimensions
        n_cols = min(3, n_methods + 1)  # +1 for consensus
        n_rows = (n_methods + 2) // n_cols  # +1 for consensus, +1 for rounding

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        plot_idx = 0

        # Plot individual detection methods
        for method_name, predictions in anomaly_results['individual_methods'].items():
            if plot_idx >= n_rows * n_cols:
                break

            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = axes[row, col]

            # Color points by anomaly status
            colors = ['red' if pred == -1 else 'blue' for pred in predictions]

            if pca_data.shape[1] >= 2:
                scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.7, s=80)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            else:
                ax.scatter(range(len(predictions)), predictions, c=colors, alpha=0.7, s=80)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Anomaly Score')

            n_anomalies = (np.array(predictions) == -1).sum()
            ax.set_title(f'{method_name.replace(\"_\", \" \").title()}\\n({n_anomalies} anomalies)',
                        fontweight='bold')
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Plot consensus results
        if 'consensus' in anomaly_results and plot_idx < n_rows * n_cols:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = axes[row, col]

            consensus = anomaly_results['consensus']
            colors = ['red' if pred == -1 else 'blue' for pred in consensus]

            if pca_data.shape[1] >= 2:
                ax.scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.7, s=80)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')

            n_consensus = (consensus == -1).sum()
            ax.set_title(f'Consensus Anomalies\\n({n_consensus} detected)', fontweight='bold')
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/anomaly_detection_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_temporal_analysis_suite(self, df: pd.DataFrame, save_path: str = None) -> None:
        """
        Create comprehensive temporal analysis visualizations
        """
        print("Creating temporal analysis suite...")

        # Find temporal columns
        time_cols = [col for col in df.columns if any(word in col.lower()
                    for word in ['time', 'month', 'date', 'follow'])]
        value_cols = [col for col in df.columns if any(word in col.lower()
                     for word in ['endpoint', 'value', 'efficacy', 'outcome'])]

        if not time_cols or not value_cols:
            print("No temporal columns found for analysis")
            return

        # Use the first available time and value columns
        time_col = time_cols[0]
        value_col = value_cols[0]

        temporal_df = df[[time_col, value_col]].dropna()

        if len(temporal_df) < 3:
            print("Insufficient temporal data for visualization")
            return

        # Create temporal visualization suite
        fig = plt.figure(figsize=(20, 15))

        # 1. Time series plot with trend line
        ax1 = plt.subplot(3, 3, 1)
        conditions = df['condition'].unique() if 'condition' in df.columns else ['All']
        colors = plt.cm.Set1(np.linspace(0, 1, len(conditions)))

        for i, condition in enumerate(conditions):
            if 'condition' in df.columns:
                condition_data = df[df['condition'] == condition][[time_col, value_col]].dropna()
            else:
                condition_data = temporal_df

            if len(condition_data) > 0:
                ax1.scatter(condition_data[time_col], condition_data[value_col],
                           label=condition, color=colors[i], alpha=0.7, s=80)

        # Add overall trend line
        if len(temporal_df) > 1:
            z = np.polyfit(temporal_df[time_col], temporal_df[value_col], 1)
            p = np.poly1d(z)
            ax1.plot(temporal_df[time_col], p(temporal_df[time_col]),
                    \"r--\", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')

        ax1.set_xlabel(time_col)
        ax1.set_ylabel(value_col)
        ax1.set_title('Temporal Progression by Condition')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribution of follow-up durations
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(temporal_df[time_col], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel(time_col)
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Follow-up Durations')
        ax2.grid(True, alpha=0.3)

        # 3. Box plot by time bins
        ax3 = plt.subplot(3, 3, 3)
        temporal_df['time_bin'] = pd.cut(temporal_df[time_col], bins=5, labels=False)
        temporal_df.boxplot(column=value_col, by='time_bin', ax=ax3)
        ax3.set_title('Outcome Distribution by Time Bins')
        ax3.set_xlabel('Time Bin')
        ax3.set_ylabel(value_col)

        # 4. Correlation analysis over time windows
        ax4 = plt.subplot(3, 3, 4)
        if len(temporal_df) > 5:
            # Rolling correlation analysis (if we have another variable)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 2:
                other_col = [col for col in numeric_cols if col not in [time_col, value_col]][0]
                temporal_with_other = df[[time_col, value_col, other_col]].dropna()

                if len(temporal_with_other) > 3:
                    # Sort by time and calculate rolling correlation
                    temporal_with_other = temporal_with_other.sort_values(time_col)
                    window_size = max(3, len(temporal_with_other) // 3)

                    rolling_corr = []
                    time_points = []

                    for i in range(window_size, len(temporal_with_other)):
                        window_data = temporal_with_other.iloc[i-window_size:i]
                        corr, _ = stats.pearsonr(window_data[value_col], window_data[other_col])
                        rolling_corr.append(corr)
                        time_points.append(window_data[time_col].iloc[-1])

                    ax4.plot(time_points, rolling_corr, 'o-', linewidth=2, markersize=6)
                    ax4.set_xlabel(time_col)
                    ax4.set_ylabel(f'Correlation with {other_col}')
                    ax4.set_title('Rolling Correlation Analysis')
                    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data\\nfor rolling correlation',
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'No additional variables\\nfor correlation analysis',
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\\nfor correlation analysis',
                    ha='center', va='center', transform=ax4.transAxes)

        # 5. Rate of change analysis
        ax5 = plt.subplot(3, 3, 5)
        temporal_sorted = temporal_df.sort_values(time_col)
        if len(temporal_sorted) > 1:
            time_diffs = np.diff(temporal_sorted[time_col])
            value_diffs = np.diff(temporal_sorted[value_col])
            rates = np.where(time_diffs != 0, value_diffs / time_diffs, 0)

            ax5.plot(temporal_sorted[time_col].iloc[1:], rates, 'o-', linewidth=2, markersize=6)
            ax5.set_xlabel(time_col)
            ax5.set_ylabel(f'Rate of Change ({value_col}/time)')
            ax5.set_title('Rate of Change Analysis')
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax5.grid(True, alpha=0.3)

        # 6. Outcome prediction based on duration
        ax6 = plt.subplot(3, 3, 6)
        if len(temporal_df) > 2:
            # Simple linear regression visualization
            from sklearn.linear_model import LinearRegression

            X = temporal_df[time_col].values.reshape(-1, 1)
            y = temporal_df[value_col].values

            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)

            ax6.scatter(temporal_df[time_col], temporal_df[value_col], alpha=0.7, s=80, label='Actual')
            ax6.plot(temporal_df[time_col], y_pred, 'r-', linewidth=2, label=f'Prediction (R²={reg.score(X, y):.3f})')
            ax6.set_xlabel(time_col)
            ax6.set_ylabel(value_col)
            ax6.set_title('Linear Prediction Model')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Residual analysis
        ax7 = plt.subplot(3, 3, 7)
        if len(temporal_df) > 2:
            residuals = y - y_pred
            ax7.scatter(y_pred, residuals, alpha=0.7, s=80)
            ax7.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax7.set_xlabel('Predicted Values')
            ax7.set_ylabel('Residuals')
            ax7.set_title('Residual Analysis')
            ax7.grid(True, alpha=0.3)

        # 8. Time-based clustering
        ax8 = plt.subplot(3, 3, 8)
        if len(temporal_df) > 3:
            from sklearn.cluster import KMeans

            # Cluster based on time and value
            X_cluster = temporal_df[[time_col, value_col]].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            n_clusters = min(3, len(temporal_df) // 2)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)

                scatter = ax8.scatter(temporal_df[time_col], temporal_df[value_col],
                                    c=clusters, cmap='viridis', alpha=0.7, s=80)
                ax8.set_xlabel(time_col)
                ax8.set_ylabel(value_col)
                ax8.set_title(f'K-Means Clustering (k={n_clusters})')
                plt.colorbar(scatter, ax=ax8, label='Cluster')
                ax8.grid(True, alpha=0.3)

        # 9. Statistical summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        # Calculate statistics
        corr_coef, p_value = stats.pearsonr(temporal_df[time_col], temporal_df[value_col])
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(temporal_df[time_col], temporal_df[value_col])

        stats_text = f\"\"\"Temporal Analysis Statistics:\n\nPearson Correlation: {corr_coef:.3f}\np-value: {p_value:.3e}\n\nLinear Regression:\nSlope: {slope:.3f}\nR²: {r_value**2:.3f}\nStd Error: {std_err:.3f}\n\nData Points: {len(temporal_df)}\nTime Range: {temporal_df[time_col].min():.1f} - {temporal_df[time_col].max():.1f}\nValue Range: {temporal_df[value_col].min():.1f} - {temporal_df[value_col].max():.1f}\"\"\"\n        \n        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=10,\n                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))\n        \n        plt.suptitle('Comprehensive Temporal Analysis Suite', fontsize=16, fontweight='bold')\n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(f'{save_path}/temporal_analysis_suite.png', dpi=300, bbox_inches='tight')\n        plt.show()\n\n    def create_interactive_correlation_network(self, correlation_data: list, \n                                             save_path: str = None) -> go.Figure:\n        \"\"\"\n        Create interactive network visualization of correlations\n        \"\"\"\n        print(\"Creating interactive correlation network...\")\n        \n        if not correlation_data:\n            print(\"No correlation data available for network visualization\")\n            return None\n        \n        # Build network graph\n        G = nx.Graph()\n        \n        # Add edges for significant correlations\n        for corr_info in correlation_data:\n            if abs(corr_info['correlation']) >= 0.4:  # Threshold for visualization\n                G.add_edge(\n                    corr_info['variable_1'],\n                    corr_info['variable_2'],\n                    weight=abs(corr_info['correlation']),\n                    correlation=corr_info['correlation'],\n                    p_value=corr_info['p_value']\n                )\n        \n        if len(G.nodes()) == 0:\n            print(\"No significant correlations found for network visualization\")\n            return None\n        \n        # Calculate layout\n        pos = nx.spring_layout(G, k=1, iterations=50)\n        \n        # Extract edges\n        edge_x = []\n        edge_y = []\n        edge_info = []\n        \n        for edge in G.edges(data=True):\n            x0, y0 = pos[edge[0]]\n            x1, y1 = pos[edge[1]]\n            edge_x.extend([x0, x1, None])\n            edge_y.extend([y0, y1, None])\n            edge_info.append(f\"{edge[0]} ↔ {edge[1]}: r={edge[2]['correlation']:.3f}\")\n        \n        # Create edge trace\n        edge_trace = go.Scatter(\n            x=edge_x, y=edge_y,\n            line=dict(width=2, color='rgba(125,125,125,0.5)'),\n            hoverinfo='none',\n            mode='lines'\n        )\n        \n        # Extract nodes\n        node_x = []\n        node_y = []\n        node_text = []\n        node_adjacencies = []\n        \n        for node in G.nodes():\n            x, y = pos[node]\n            node_x.append(x)\n            node_y.append(y)\n            node_text.append(node)\n            node_adjacencies.append(len(list(G.neighbors(node))))\n        \n        # Create node trace\n        node_trace = go.Scatter(\n            x=node_x, y=node_y,\n            mode='markers+text',\n            hoverinfo='text',\n            text=node_text,\n            textposition=\"middle center\",\n            marker=dict(\n                size=[adj*10 + 20 for adj in node_adjacencies],\n                color=node_adjacencies,\n                colorscale='Viridis',\n                colorbar=dict(\n                    thickness=15,\n                    len=0.5,\n                    x=1.1,\n                    title=\"Node Connections\"\n                ),\n                line=dict(width=2, color='white')\n            )\n        )\n        \n        # Create figure\n        fig = go.Figure(data=[edge_trace, node_trace],\n                       layout=go.Layout(\n                           title='Interactive Correlation Network<br>Node size = number of connections',\n                           titlefont_size=16,\n                           showlegend=False,\n                           hovermode='closest',\n                           margin=dict(b=20,l=5,r=5,t=40),\n                           annotations=[ dict(\n                               text=\"Variables with significant correlations are connected. Hover over nodes for details.\",\n                               showarrow=False,\n                               xref=\"paper\", yref=\"paper\",\n                               x=0.005, y=-0.002,\n                               xanchor='left', yanchor='bottom',\n                               font=dict(color=\"gray\", size=12)\n                           )],\n                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),\n                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),\n                           width=1000,\n                           height=800\n                       ))\n        \n        if save_path:\n            fig.write_html(f'{save_path}/interactive_correlation_network.html')\n        \n        return fig\n\n    def create_dimensional_reduction_suite(self, df: pd.DataFrame, save_path: str = None) -> None:\n        \"\"\"\n        Create comprehensive dimensional reduction visualizations\n        \"\"\"\n        print(\"Creating dimensional reduction visualization suite...\")\n        \n        # Prepare numeric data\n        numeric_df = df.select_dtypes(include=[np.number]).dropna()\n        \n        if len(numeric_df.columns) < 3 or len(numeric_df) < 3:\n            print(\"Insufficient data for dimensional reduction\")\n            return\n        \n        # Standardize data\n        scaler = StandardScaler()\n        scaled_data = scaler.fit_transform(numeric_df)\n        \n        # Create visualization suite\n        fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n        \n        # 1. PCA 2D\n        pca_2d = PCA(n_components=2)\n        pca_2d_data = pca_2d.fit_transform(scaled_data)\n        \n        scatter = axes[0, 0].scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], \n                                   c=range(len(pca_2d_data)), cmap='viridis', alpha=0.7, s=80)\n        axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')\n        axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')\n        axes[0, 0].set_title('PCA 2D Projection')\n        axes[0, 0].grid(True, alpha=0.3)\n        plt.colorbar(scatter, ax=axes[0, 0], label='Sample Index')\n        \n        # 2. PCA 3D preparation and biplot\n        if scaled_data.shape[1] >= 3:\n            pca_3d = PCA(n_components=3)\n            pca_3d_data = pca_3d.fit_transform(scaled_data)\n            \n            # Create biplot\n            axes[0, 1].scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], alpha=0.6, s=80)\n            \n            # Add loading vectors\n            loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)\n            \n            for i, (feature, loading) in enumerate(zip(numeric_df.columns, loadings)):\n                axes[0, 1].arrow(0, 0, loading[0]*3, loading[1]*3, \n                               head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)\n                axes[0, 1].text(loading[0]*3.3, loading[1]*3.3, feature, \n                               fontsize=8, ha='center', va='center')\n            \n            axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')\n            axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')\n            axes[0, 1].set_title('PCA Biplot with Feature Loadings')\n            axes[0, 1].grid(True, alpha=0.3)\n        \n        # 3. t-SNE if enough data\n        if len(scaled_data) >= 5 and scaled_data.shape[1] >= 2:\n            try:\n                tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(scaled_data)-1))\n                tsne_data = tsne.fit_transform(scaled_data)\n                \n                scatter = axes[0, 2].scatter(tsne_data[:, 0], tsne_data[:, 1], \n                                           c=range(len(tsne_data)), cmap='plasma', alpha=0.7, s=80)\n                axes[0, 2].set_xlabel('t-SNE 1')\n                axes[0, 2].set_ylabel('t-SNE 2')\n                axes[0, 2].set_title('t-SNE 2D Projection')\n                axes[0, 2].grid(True, alpha=0.3)\n                plt.colorbar(scatter, ax=axes[0, 2], label='Sample Index')\n            except Exception as e:\n                axes[0, 2].text(0.5, 0.5, f't-SNE failed:\\n{str(e)[:50]}...', \n                               ha='center', va='center', transform=axes[0, 2].transAxes)\n        else:\n            axes[0, 2].text(0.5, 0.5, 'Insufficient data\\nfor t-SNE', \n                           ha='center', va='center', transform=axes[0, 2].transAxes)\n        \n        # 4. Scree plot\n        pca_full = PCA()\n        pca_full.fit(scaled_data)\n        \n        axes[1, 0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1), \n                       pca_full.explained_variance_ratio_, 'bo-', linewidth=2, markersize=8)\n        axes[1, 0].set_xlabel('Principal Component')\n        axes[1, 0].set_ylabel('Explained Variance Ratio')\n        axes[1, 0].set_title('Scree Plot')\n        axes[1, 0].grid(True, alpha=0.3)\n        \n        # 5. Cumulative variance\n        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)\n        axes[1, 1].plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'ro-', linewidth=2, markersize=8)\n        axes[1, 1].axhline(y=0.8, color='k', linestyle='--', alpha=0.5, label='80%')\n        axes[1, 1].axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95%')\n        axes[1, 1].set_xlabel('Number of Components')\n        axes[1, 1].set_ylabel('Cumulative Explained Variance')\n        axes[1, 1].set_title('Cumulative Explained Variance')\n        axes[1, 1].legend()\n        axes[1, 1].grid(True, alpha=0.3)\n        \n        # 6. Feature importance in first 2 PCs\n        feature_importance = np.abs(pca_2d.components_).sum(axis=0)\n        sorted_idx = np.argsort(feature_importance)[::-1]\n        \n        axes[1, 2].bar(range(len(feature_importance)), feature_importance[sorted_idx])\n        axes[1, 2].set_xlabel('Features')\n        axes[1, 2].set_ylabel('Importance in PC1+PC2')\n        axes[1, 2].set_title('Feature Importance in Principal Components')\n        axes[1, 2].set_xticks(range(len(feature_importance)))\n        axes[1, 2].set_xticklabels([numeric_df.columns[i] for i in sorted_idx], rotation=45)\n        axes[1, 2].grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        if save_path:\n            plt.savefig(f'{save_path}/dimensional_reduction_suite.png', dpi=300, bbox_inches='tight')\n        plt.show()\n\n    def generate_comprehensive_report(self, df: pd.DataFrame, \n                                    correlation_data: list = None,\n                                    anomaly_results: dict = None,\n                                    save_path: str = '../results') -> None:\n        \"\"\"\n        Generate comprehensive visualization report\n        \"\"\"\n        print(\"\\n\" + \"=\"*80)\n        print(\"GENERATING COMPREHENSIVE VISUALIZATION SUITE\")\n        print(\"=\"*80)\n        \n        # Create results directory if it doesn't exist\n        import os\n        os.makedirs(save_path, exist_ok=True)\n        \n        # 1. Correlation Analysis\n        print(\"\\n1. Creating correlation heatmap suite...\")\n        self.create_correlation_heatmap_suite(df, save_path)\n        \n        # 2. Anomaly Detection Visualizations\n        if anomaly_results:\n            print(\"\\n2. Creating anomaly detection plots...\")\n            self.create_anomaly_detection_plots(df, anomaly_results, save_path)\n        \n        # 3. Temporal Analysis\n        print(\"\\n3. Creating temporal analysis suite...\")\n        self.create_temporal_analysis_suite(df, save_path)\n        \n        # 4. Interactive Network\n        if correlation_data:\n            print(\"\\n4. Creating interactive correlation network...\")\n            interactive_fig = self.create_interactive_correlation_network(correlation_data, save_path)\n            if interactive_fig:\n                interactive_fig.show()\n        \n        # 5. Dimensional Reduction\n        print(\"\\n5. Creating dimensional reduction suite...\")\n        self.create_dimensional_reduction_suite(df, save_path)\n        \n        print(\"\\n\" + \"=\"*80)\n        print(\"VISUALIZATION SUITE GENERATION COMPLETE\")\n        print(f\"All visualizations saved to: {save_path}/\")\n        print(\"Generated files:\")\n        print(\"  • correlation_heatmap_suite.png\")\n        print(\"  • anomaly_detection_plots.png (if anomaly data provided)\")\n        print(\"  • temporal_analysis_suite.png\")\n        print(\"  • interactive_correlation_network.html (if correlation data provided)\")\n        print(\"  • dimensional_reduction_suite.png\")\n        print(\"=\"*80)\n\n\ndef main():\n    \"\"\"Main function to demonstrate visualization engine\"\"\"\n    print(\"Loading data for comprehensive visualization...\")\n    \n    # Load data\n    df = pd.read_csv('../data/clinical_trial_data.csv')\n    \n    # Initialize visualization engine\n    viz_engine = StatisticalVisualizationEngine()\n    \n    # Generate comprehensive report\n    viz_engine.generate_comprehensive_report(df)\n    \n    print(\"Visualization suite generation completed!\")\n\n\nif __name__ == \"__main__\":\n    main()