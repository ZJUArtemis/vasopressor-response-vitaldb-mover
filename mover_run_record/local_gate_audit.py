"""Streaming real-source G1-G3 audit; never exports patient-level rows."""
import tarfile,json,logging,re,sys,hashlib,io
from pathlib import Path
from collections import Counter
import pandas as pd
class ForwardReader(io.RawIOBase):
    def __init__(self,f): self.f=f
    def readable(self): return True
    def readinto(self,b):
        d=self.f.read(len(b)); b[:len(d)]=d; return len(d)
def buffered(f): return io.BufferedReader(ForwardReader(f))
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'logs/local_gate_audit.log'),logging.StreamHandler()])
archive=Path('<MOVER_ARCHIVE>/EPIC_EMR.tar.gz')
actions={}; category=None
for line in (ROOT/'config/mover_mar_actions.yaml').read_text().splitlines():
    if line and not line.startswith(('#',' ')): category=line.split(':')[0]
    m=re.match(r'\s*-\s*"([^"]+)"',line)
    if m: actions[m[1]]=category
patterns={'phenylephrine':['phenylephrine','neo-synephrine','neosynephrine'],'ephedrine':['ephedrine'],'norepinephrine':['norepinephrine','levophed'],'epinephrine':['epinephrine','adrenaline'],'vasopressin':['vasopressin','pitressin']}
counts=Counter(); vocab=Counter(); drugs=Counter(); units=Counter(); perdrug={}
with tarfile.open(archive,'r|gz') as tar:
    for member in tar:
        if member.name.endswith('/patient_information.csv'):
            p=pd.read_csv(buffered(tar.extractfile(member)),dtype=str,keep_default_na=False)
            wi=pd.to_datetime(p.IN_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce'); wo=pd.to_datetime(p.OUT_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce')
            windows=dict(zip(p.LOG_ID,zip(wi,wo)))
            logging.info('Loaded %d surgery windows',len(windows))
        elif member.name.endswith('/patient_medications.csv'):
            # Windows may occur later in archive; use a separate short streaming pass if needed.
            if 'windows' not in locals():
                with tarfile.open(archive,'r:gz') as t2:
                    pm=next(m for m in t2 if m.name.endswith('/patient_information.csv'))
                    p=pd.read_csv(t2.extractfile(pm),dtype=str,keep_default_na=False)
                    windows=dict(zip(p.LOG_ID,zip(pd.to_datetime(p.IN_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce'),pd.to_datetime(p.OUT_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce'))))
            for i,c in enumerate(pd.read_csv(buffered(tar.extractfile(member)),dtype=str,keep_default_na=False,chunksize=200000)):
                counts['all_med_rows']+=len(c)
                name=c.MEDICATION_NM.str.lower(); drug=pd.Series('',index=c.index)
                for d,ps in patterns.items():
                    mask=name.str.contains('|'.join(map(re.escape,ps)),regex=True)&drug.eq(''); drug.loc[mask]=d
                c=c[drug.ne('')].copy(); c['_drug']=drug.loc[c.index]; counts['vaso_rows']+=len(c)
                vocab.update(c.MAR_ACTION_NM)
                s=c[c.MAR_ACTION_NM.map(actions).eq('keep_as_administration')].copy()
                t=pd.to_datetime(s.MED_ACTION_TIME,format='ISO8601',errors='coerce')
                ins=s.LOG_ID.map(lambda k:windows.get(k,(pd.NaT,pd.NaT))[0]); outs=s.LOG_ID.map(lambda k:windows.get(k,(pd.NaT,pd.NaT))[1])
                inwin=t.ge(ins)&t.le(outs)
                dose=pd.to_numeric(s.ADMIN_SIG,errors='coerce'); unit=s.DOSE_UNIT_NM.str.strip().ne('')
                for scope,mask in [('all',pd.Series(True,index=s.index)),('in_window',inwin)]:
                    counts[scope+'_n']+=int(mask.sum()); counts[scope+'_time']+=int((mask&t.notna()).sum()); counts[scope+'_dose']+=int((mask&dose.notna()).sum()); counts[scope+'_dose_unit']+=int((mask&dose.notna()&unit).sum())
                counts['unparseable_start_time']+=int(t.isna().sum())
                drugs.update(s.loc[inwin,'_drug']); units.update(s.loc[inwin,'DOSE_UNIT_NM'].replace('','<missing>'))
                for d,g in s.loc[inwin].groupby('_drug'):
                    a=perdrug.setdefault(d,Counter()); a['n']+=len(g); a['dose_numeric']+=int(dose.loc[g.index].notna().sum()); a['dose_and_unit']+=int((dose.loc[g.index].notna()&unit.loc[g.index]).sum())
                if i%10==0:logging.info('Scanned %d MAR rows',counts['all_med_rows'])
            break
n=counts['in_window_n']; unknown={k:v for k,v in vocab.items() if k not in actions}
g1=counts['all_time']/max(counts['all_n'],1); g2=counts['in_window_dose_unit']/max(n,1)
r={'analysis_scope':'real MOVER; streaming original archive; no synthetic rows','counts':dict(counts),'n_vaso_mar_rows':counts['vaso_rows'],'n_start_admin':n,'by_drug_start':dict(drugs),'G1_timestamp_parse_frac':g1,'G1_pass':g1>=.9,'G1_denominator':'all matched administration rows (includes unparseable timestamps); intraoperative eligibility evaluated separately','G2_dose_parse_frac':counts['in_window_dose']/max(n,1),'G2_dose_and_unit_frac':g2,'G2_pass':g2>=.8,'G3_action_vocab_fully_mapped':not unknown,'G3_unmapped_actions':unknown,'action_counts':dict(vocab),'dose_units_in_window':dict(units),'per_drug':{k:dict(v) for k,v in perdrug.items()},'G4_status':'NOT_RUN; requires G1-G3 GO','source_columns':{'med_time_col':'MED_ACTION_TIME','med_action_col':'MAR_ACTION_NM','med_dose_col':'ADMIN_SIG','med_dose_unit_col':'DOSE_UNIT_NM'}}
r['VERDICT']='GO' if r['G1_pass'] and r['G2_pass'] and not unknown else 'PAUSE'
(OUT/'m0_dose_audit.json').write_text(json.dumps(r,indent=2))
logging.info('VERDICT %s; G1 %.4f; G2 %.4f; unmapped %s',r['VERDICT'],g1,g2,unknown)
