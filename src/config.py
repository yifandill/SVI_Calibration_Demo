import os
import re
from datetime import datetime
import io
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'dataset')
result_path = os.path.join(script_dir, '..', 'results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
dates = []
for filename in os.listdir(data_path):
    match = date_pattern.search(filename)
    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        dates.append(date_str)

# Context manager to suppress stdout
class SuppressOutput:
    def __init__(self, suppress_stdout=True, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = io.StringIO()

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        
        if self.suppress_stderr:
            sys.stderr = self._stderr
