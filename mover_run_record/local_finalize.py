"""Finalize full return directory; user confirmed DUA permits event-level transfer."""
from pathlib import Path
import json,hashlib,shutil,tarfile,gzip
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';LOCAL=OUT
# Losslessly compress large console logs; retain every warning and log line.
compressed_logs={}
for existing in (OUT/'logs').glob('*.log.gz'):
    digest=hashlib.sha256();size=0
    with gzip.open(existing,'rb') as source:
        while chunk:=source.read(1<<20):
            digest.update(chunk);size+=len(chunk)
    compressed_logs[str(existing.relative_to(OUT))]={'uncompressed_bytes':size,'uncompressed_sha256':digest.hexdigest()}

for log in (OUT/'logs').glob('*.log'):
    if log.stat().st_size > 10_000_000:
        with log.open('rb') as source:
            digest=hashlib.file_digest(source,'sha256').hexdigest()
        size=log.stat().st_size
        target=log.with_suffix(log.suffix+'.gz')
        with log.open('rb') as source, gzip.open(target,'wb') as dest:
            shutil.copyfileobj(source,dest)
        compressed_logs[str(target.relative_to(OUT))]={'uncompressed_bytes':size,'uncompressed_sha256':digest}
        log.unlink()
# User explicitly confirmed event-level transfer in this chat.
ev=pd.read_parquet(LOCAL/'mover_documented_events.parquet');st=ev[ev.event_type.eq('start')].copy()
summary=st.groupby('drug').agg(n_start=('drug','size'),n_cases=('caseid','nunique'),n_patients=('subjectid','nunique'),baseline_map_median=('baseline_map','median'),delta_map_5min_mean=('delta_map_5min','mean'),delta_map_5min_median=('delta_map_5min','median'),same_drug_redose_frac=('censored_repeat','mean'),dose_missing_frac=('dose_rate_at_event',lambda s:s.isna().mean()))
summary.to_csv(OUT/'m1_event_summary.csv')
st.groupby(['drug','dose_unit','med_route'],dropna=False).size().rename('n_start').reset_index().to_csv(OUT/'m1_event_dose_unit_route_counts.csv',index=False)
rep=OUT/'reproducibility';rep.mkdir(exist_ok=True)
for p in (ROOT/'src').glob('local_*.py'):shutil.copy2(p,rep/p.name)
for p in [ROOT/'config/mover_columns_local.yaml',ROOT/'config/mover_mar_actions.yaml',ROOT/'TRANSFER_AND_RETURN.md']:shutil.copy2(p,rep/p.name)
manifest={'analysis':'Topic 2 v3 real MOVER','transfer_policy':'full outputs including per-event physiology and case/patient identifiers; user explicitly confirmed DUA permits transfer','event_files':[p.name for p in sorted(OUT.glob('*.parquet'))],'source_archives':[],'files':{},'lossless_log_compression':compressed_logs}
for p in [Path('<MOVER_ARCHIVE>/EPIC_EMR.tar.gz'),Path('<MOVER_ARCHIVE>/Epic_flowsheets_cleaned.tar.gz')]:
 s=p.stat();manifest['source_archives'].append({'name':p.name,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'hash_status':'not hashed; source size/time and complete member scan recorded'})
for p in [ROOT.parent/'topic2_v3_mover_package.tar.gz']+list((ROOT/'models/vitaldb').glob('*.pkl')):
 manifest.setdefault('input_sha256',{})[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name not in ['RETURN_MANIFEST.json','CHECKSUMS_SHA256.txt']:
  manifest['files'][str(p.relative_to(OUT))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/'RETURN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
checks=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name!='CHECKSUMS_SHA256.txt': checks.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT)))
(OUT/'CHECKSUMS_SHA256.txt').write_text('\n'.join(checks)+'\n')
archive=ROOT.parent/'topic2_v3_mover_outputs.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(OUT,arcname='outputs')
print(archive)
