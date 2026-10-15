import subprocess,json,sys
from pathlib import Path
p=Path(__file__).parent;log=p/'smoke.log'
with log.open('w') as f:
 r=subprocess.run([sys.executable,str(p/'src/m0_dose_audit.py'),'--config',str(p/'fixtures/mover_columns.yaml')],stdout=f,stderr=subprocess.STDOUT);assert r.returncode==0
 r=subprocess.run([sys.executable,str(p/'src/local_extract_events.py'),'--config',str(p/'fixtures/mover_columns.yaml')],stdout=f,stderr=subprocess.STDOUT)
 gate=json.loads((p/'outputs/m0_dose_audit.json').read_text());assert r.returncode!=0 and not gate['G4_pass']
 gate['G4_pass']=True;gate['synthetic_model_smoke_override']='ONLY synthetic smoke; real G4 unchanged'
 (p/'outputs/m0_dose_audit.json').write_text(json.dumps(gate))
 r=subprocess.run([sys.executable,str(p/'src/local_model_and_external.py'),'--min_n','5'],stdout=f,stderr=subprocess.STDOUT);assert r.returncode==0
import pandas as pd
n=pd.read_csv(p/'outputs/m2_mover_native.csv');e=pd.read_csv(p/'outputs/m2_external_validation.csv')
assert len(n)==9 and len(e)==6
assert set(n.drug)=={'ephedrine','phenylephrine','norepinephrine'}
assert set(e.drug)=={'phenylephrine','norepinephrine'}
print(json.dumps({'synthetic_smoke':'PASS','G4_stop_rule_exercised':True,'native_model_rows':len(n),'external_validation_rows':len(e),'real_data_results_mixed':False}))
