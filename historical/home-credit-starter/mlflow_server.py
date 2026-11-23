"""
Start MLflow tracking UI.
Run: python mlflow_server.py
Then open: http://localhost:5000
"""
import subprocess
import sys

if __name__ == '__main__':
    print('Starting MLflow UI at http://localhost:5000 ...')
    print('Press Ctrl+C to stop.\n')
    subprocess.run([sys.executable, '-m', 'mlflow', 'ui', '--host', '0.0.0.0', '--port', '5000'])
