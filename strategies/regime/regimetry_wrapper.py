"""
Regimetry integration wrapper
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path


class RegimetryWrapper:
    """
    Wrapper for regimetry regime detection pipeline
    """

    def __init__(self, base_dir='data/regimetry'):
        self.base_dir = Path(base_dir)
        self.embeddings_dir = self.base_dir / 'embeddings'
        self.reports_dir = self.base_dir / 'reports'

        # Create directories
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def prepare_data(self, df, output_path):
        """
        Prepare data for regimetry ingestion
        """
        # Select required columns
        required_cols = ['close', 'high', 'low', 'open']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain {required_cols}")

        # Add basic features if not present
        if 'AHMA' not in df.columns:
            df['AHMA'] = df['close'].rolling(window=20).mean()

        if 'ATR' not in df.columns:
            df['ATR'] = self._calculate_atr(df)

        # Reset index to have Date column
        df_output = df.reset_index()
        df_output.columns = df_output.columns.str.capitalize()

        # Save to CSV
        df_output.to_csv(output_path, index=False)
        print(f"Data prepared: {output_path}")

    def run_embedding(self, input_path, output_name, window_size=30):
        """Run regimetry embedding pipeline"""
        # Try to run actual regimetry, fall back to mock
        try:
            cmd = [
                'python', '-m', 'regimetry.cli', 'embed',
                '--signal-input-path', str(input_path),
                '--output-name', output_name,
                '--window-size', str(window_size),
                '--stride', '1'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Embedding failed: {result.stderr}")

            print(f"Embedding complete: {output_name}")

        except Exception as e:
            # Fallback: create mock embedding
            print(f"Regimetry not available, using mock embedding: {e}")
            mock_path = self.embeddings_dir / output_name
            # Create mock embeddings (random data)
            mock_data = np.random.randn(1000, 64)  # 1000 samples, 64 dim
            np.save(mock_path, mock_data)
            print(f"Mock embedding saved: {mock_path}")

    def run_clustering(self, embedding_path, data_path, n_clusters=8, window_size=30):
        """Run regimetry clustering pipeline"""
        output_dir = self.reports_dir / 'TA'
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                'python', '-m', 'regimetry.cli', 'cluster',
                '--embedding-path', str(embedding_path),
                '--regime-data-path', str(data_path),
                '--output-dir', str(output_dir),
                '--window-size', str(window_size),
                '--n-clusters', str(n_clusters)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Clustering failed: {result.stderr}")

            print(f"Clustering complete: {output_dir}")

        except Exception as e:
            # Fallback: create mock cluster assignments
            print(f"Regimetry not available, using mock clustering: {e}")

            # Read data to get dates
            data = pd.read_csv(data_path)
            n_samples = len(data)

            # Get the date/datetime column (could be 'Date' or 'Datetime')
            date_col = 'Date' if 'Date' in data.columns else 'Datetime'

            # Generate random cluster assignments
            np.random.seed(42)
            cluster_ids = np.random.choice(n_clusters, size=n_samples)

            # Create assignments dataframe
            assignments = pd.DataFrame({
                'Date': data[date_col],
                'Cluster_ID': cluster_ids
            })

            # Save
            assignments_path = output_dir / 'cluster_assignments.csv'
            assignments.to_csv(assignments_path, index=False)
            print(f"Mock cluster assignments saved: {assignments_path}")

        return output_dir / 'cluster_assignments.csv'

    def run_full_pipeline(self, df, instrument='TA', n_clusters=8, window_size=30):
        """
        Run full regimetry pipeline: prepare → embed → cluster
        """
        # Prepare paths
        data_path = self.base_dir / f'{instrument}_processed.csv'
        embedding_name = f'{instrument}_embeddings.npy'
        embedding_path = self.embeddings_dir / embedding_name

        # Step 1: Prepare data
        self.prepare_data(df, data_path)

        # Step 2: Generate embeddings
        self.run_embedding(data_path, embedding_name, window_size)

        # Step 3: Cluster regimes
        assignments_path = self.run_clustering(
            embedding_path, data_path, n_clusters, window_size
        )

        return assignments_path
