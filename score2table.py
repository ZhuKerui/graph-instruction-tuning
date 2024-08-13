import sys
from collections import defaultdict
import pandas as pd

results = defaultdict(list)
with open(sys.argv[1]) as f_in:
    for line in f_in:
        if line.startswith('..'):
            results['setting'].append(line.strip())
        else:
            task, scores = line.strip().split(':')
            task = task.split('(')[0].strip()
            if 'F1' in scores:
                score = float(scores.split('F1 ')[1].split()[0])
                task += ' F1'
            else:
                score = float(scores.split('MSE')[0].split('EM ')[1].split()[0])
                task += ' EM'
            results[task].append(score)

pd.DataFrame(results).to_csv(sys.argv[2], index=False)
