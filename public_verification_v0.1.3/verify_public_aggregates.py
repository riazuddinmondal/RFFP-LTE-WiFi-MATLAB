#!/usr/bin/env python3
"""Recalculate exactly what public sufficient statistics allow; no private data needed."""
import csv,math,collections,sys
from pathlib import Path
root=Path(__file__).resolve().parent
read=lambda f:list(csv.DictReader(open(root/f,encoding='utf-8-sig',newline='')))
folds=read('tenfold_fold_sufficient_statistics.csv'); pooled={r['Method']:r for r in read('tenfold_pooled_metrics.csv')}; counts=read('tenfold_threshold_counts_by_fold.csv'); comp=read('fold_campaign_device_composition.csv'); integrity=read('fold_integrity_aggregate.csv')
by=collections.defaultdict(list)
for r in folds:by[r['Method']].append(r)
assert len(by)==8 and all(len(rows)==10 for rows in by.values())
for method,rows in by.items():
 n=sum(int(r['N']) for r in rows);assert n==118930,(method,n)
 s=sum(float(r['SumError_m']) for r in rows);s2=sum(float(r['SumSquaredError_m2']) for r in rows)
 mean=s/n;rmse=math.sqrt(s2/n);target=pooled[method]
 assert abs(mean-float(target['Mean_m']))<1e-7,(method,'mean',mean,target['Mean_m'])
 assert abs(rmse-float(target['RMSE_m']))<1e-7,(method,'RMSE',rmse,target['RMSE_m'])
 print(f'{method:24s} N={n} mean={mean:.6f} m RMSE={rmse:.6f} m  PASS')
for method in {r['Method'] for r in counts}:
 rows=[r for r in counts if r['Method']==method];assert len(rows)==10
 for r in rows:
  n=int(r['N']);values=[int(r[k]) for k in ('Within10m','Within25m','Within50m','Within100m')]
  assert 0<=values[0]<=values[1]<=values[2]<=values[3]<=n
 n=sum(int(r['N']) for r in rows);assert n==118930
 print(method,'threshold coverage:',*[f'{100*sum(int(r[k]) for r in rows)/n:.2f}%' for k in ('Within10m','Within25m','Within50m','Within100m')])
assert len(comp)==11 and comp[-1]['Fold']=='TOTAL'
for col in ('Samsung_GT_I9305_2014_Observations','Samsung_GT_I9305_2014_Logs','Samsung_SM_G900F_2015_Observations','Samsung_SM_G900F_2015_Logs','TotalObservations','TotalLogs'):
 assert sum(int(r[col]) for r in comp[:-1])==int(comp[-1][col]),col
assert int(comp[-1]['TotalObservations'])==118930 and int(comp[-1]['TotalLogs'])==38
assert next(r for r in integrity if r['Check']=='log_ids_in_multiple_outer_folds')['Value']=='0'
print('Fold composition arithmetic and published integrity assertion: PASS')
print('LIMITATION: integrity assertion is author-derived; private LogID mapping is not publicly rechecked.')
print('LIMITATION: exact P68/P95/median cannot be reconstructed from per-fold sums and aggregates.')
print('LIMITATION: aggregate-only verification does not independently validate predictions or model training.')
