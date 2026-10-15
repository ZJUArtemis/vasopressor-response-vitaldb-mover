"""Recheck G1-G3 on the complete original-source vasopressor MAR extract."""
from pathlib import Path
import json,logging
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'logs/local_recheck_gate.log'),logging.StreamHandler()])
cfg=yaml.safe_load((ROOT/'config/mover_columns_local.yaml').read_text());actions=yaml.safe_load((ROOT/'config/mover_mar_actions.yaml').read_text())
med=pd.read_csv(cfg['med_file'],dtype=str,keep_default_na=False);p=pd.read_csv(cfg['patient_file'],dtype=str,keep_default_na=False)
if len(med)!=json.loads((OUT/'m0_dose_audit_initial.json').read_text())['n_vaso_mar_rows']:raise ValueError('MAR extract completeness mismatch')
name=med.MEDICATION_NM.str.lower();med['_drug']=''
for d,pats in sorted(cfg['vaso_name_patterns'].items(), key=lambda item: item[0] != 'norepinephrine'):
 mask=name.map(lambda n:any(pat in n for pat in pats))&med['_drug'].eq('');med.loc[mask,'_drug']=d
seen=med.MAR_ACTION_NM.value_counts();known=set(sum(actions.values(),[]));unknown={k:int(v) for k,v in seen.items() if k not in known}
s=med[med.MAR_ACTION_NM.isin(actions['keep_as_administration'])].copy();t=pd.to_datetime(s.MED_ACTION_TIME,format='ISO8601',errors='coerce')
tin=dict(zip(p.LOG_ID,pd.to_datetime(p.IN_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce')));tout=dict(zip(p.LOG_ID,pd.to_datetime(p.OUT_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce')))
inwin=t.ge(s.LOG_ID.map(tin))&t.le(s.LOG_ID.map(tout));si=s.loc[inwin].copy();dose=pd.to_numeric(si.ADMIN_SIG,errors='coerce');unit=si.DOSE_UNIT_NM.str.strip().ne('')
g1=float(t.notna().mean());g2=float((dose.notna()&unit).mean())
r={'analysis_scope':'complete real-source matched MAR; unique nonconflicting valid surgery windows; no synthetic rows','n_vaso_mar_rows':len(med),'n_start_admin_all':len(s),'n_start_admin':len(si),'n_surgery_windows':len(p),'G1_timestamp_parse_frac':g1,'G1_pass':g1>=.9,'G1_denominator':'all matched START rows, retaining invalid timestamps in denominator','G1_timestamp_text_has_HH_MM_frac':float(s.MED_ACTION_TIME.str.contains(r'\d{1,2}:\d{2}').mean()),'G2_dose_parse_frac':float(dose.notna().mean()),'G2_dose_and_unit_frac':g2,'G2_pass':g2>=.8,'G2_denominator':'START administration time within unambiguous valid surgery window','G3_action_vocab_fully_mapped':not unknown,'G3_unmapped_actions':unknown,'action_counts':{k:int(v) for k,v in seen.items()},'by_drug_start':{k:int(v) for k,v in si._drug.value_counts().items()},'dose_units_in_window':{k:int(v) for k,v in si.DOSE_UNIT_NM.replace('','<missing>').value_counts().items()},'G4_status':'PENDING','action_classification_additions':{'IV Resume':'keep_as_rate_change','IV Resume Dual Sign':'keep_as_rate_change','Override pull for Anesthesia':'exclude'}}
r['VERDICT']='GO' if r['G1_pass'] and r['G2_pass'] and not unknown else 'PAUSE'
(OUT/'m0_dose_audit.json').write_text(json.dumps(r,indent=2))
si.groupby(['_drug','DOSE_UNIT_NM','MED_ROUTE_NM'],dropna=False).size().rename('n_start').reset_index().to_csv(OUT/'m0_start_dose_unit_route_counts.csv',index=False)
logging.info('G1-G3 %s; %d START; timestamp %.5f dose/unit %.5f',r['VERDICT'],len(si),g1,g2)
if r['VERDICT']!='GO':raise SystemExit(2)
