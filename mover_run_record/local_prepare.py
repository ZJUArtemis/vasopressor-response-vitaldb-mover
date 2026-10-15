"""Prepare local CSV inputs from read-only archives with bounded-memory scans."""
import tarfile,json,logging,io,re
from pathlib import Path
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]; WORK=ROOT/'local_work'; WORK.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(ROOT/'outputs/logs/local_prepare.log'),logging.StreamHandler()])
class ForwardReader(io.RawIOBase):
    def __init__(self,f): self.f=f
    def readable(self): return True
    def readinto(self,b):
        d=self.f.read(len(b));b[:len(d)]=d;return len(d)
def buffered(f): return io.BufferedReader(ForwardReader(f))
cfg=yaml.safe_load((ROOT/'config/mover_columns.yaml').read_text())
for k,v in {'surgery_table':str(WORK/'patient_information.csv'),'patient_file':str(WORK/'patient_information.csv'),'med_file':str(WORK/'vasopressor_medications.csv'),'obs_file':str(WORK/'arterial_observations.csv'),'med_time_col':'MED_ACTION_TIME','med_action_col':'MAR_ACTION_NM','med_dose_col':'ADMIN_SIG','med_dose_unit_col':'DOSE_UNIT_NM','med_route_col':'MED_ROUTE_NM','age_col':'BIRTH_DATE'}.items(): cfg[k]=v
stats={}; medparts=[]
with tarfile.open('<MOVER_ARCHIVE>/EPIC_EMR.tar.gz','r|gz') as tar:
    for m in tar:
        if m.name.endswith('/patient_information.csv'):
            p=pd.read_csv(buffered(tar.extractfile(m)),dtype=str,keep_default_na=False)
            stats['patient_rows']=len(p); stats['duplicate_LOG_ID_rows']=int(p.LOG_ID.duplicated().sum())
            p=p.drop_duplicates()
            conflict=p.LOG_ID.duplicated(keep=False)
            stats['conflicting_case_ids_excluded']=int(p.loc[conflict,'LOG_ID'].nunique())
            p=p.loc[~conflict].copy()
            tin=pd.to_datetime(p.IN_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce');tout=pd.to_datetime(p.OUT_OR_DTTM,format='%m/%d/%y %H:%M',errors='coerce')
            valid=tin.notna() & tout.notna() & tin.lt(tout)
            stats['invalid_windows_excluded']=int((~valid).sum())
            p=p.loc[valid].copy()
            stats['valid_surgery_cases']=len(p)
            # Source height is feet/inches. Source weight is ounces (existing audit); preserve missing when malformed.
            h=p.HEIGHT.str.extract(r"^(\d+)\s*'\s*(\d+(?:\.\d+)?)")
            p['HEIGHT_CM']=pd.to_numeric(h[0],errors='coerce')*30.48+pd.to_numeric(h[1],errors='coerce')*2.54
            p['WEIGHT_KG']=pd.to_numeric(p.WEIGHT,errors='coerce')*0.028349523125
            p.to_csv(WORK/'patient_information.csv',index=False)
        elif m.name.endswith('/patient_medications.csv'):
            for i,c in enumerate(pd.read_csv(buffered(tar.extractfile(m)),dtype=str,keep_default_na=False,chunksize=200000)):
                n=c.MEDICATION_NM.str.lower(); drug=pd.Series('',index=c.index)
                for d,ps in sorted(cfg['vaso_name_patterns'].items(), key=lambda item: item[0] != 'norepinephrine'):
                    mask=n.str.contains('|'.join(map(re.escape,ps)))&drug.eq('');drug.loc[mask]=d
                medparts.append(c[drug.ne('')])
                if i%20==0:logging.info('MAR extraction %d rows',(i+1)*200000)
            med=pd.concat(medparts,ignore_index=True);med.to_csv(WORK/'vasopressor_medications.csv',index=False);stats['matched_med_rows']=len(med)
            break
cfg['height_col']='HEIGHT_CM';cfg['weight_col']='WEIGHT_KG'
(ROOT/'config/mover_columns_local.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
# Keep arterial records with original timestamps; no minute aggregation or interpolation.
first=True;n=0;members=[]
with tarfile.open('<MOVER_ARCHIVE>/Epic_flowsheets_cleaned.tar.gz','r|gz') as tar:
    for m in tar:
        if not m.isfile() or not m.name.endswith('.csv'): continue
        members.append({'name':m.name,'size':m.size});logging.info('Scanning arterial observations: %s',m.name)
        for c in pd.read_csv(buffered(tar.extractfile(m)),dtype=str,keep_default_na=False,chunksize=300000,usecols=['LOG_ID','FLO_DISPLAY_NAME','MEAS_VALUE','RECORDED_TIME']):
            c=c[c.FLO_DISPLAY_NAME.isin([cfg['map_name_value'],cfg['bp_art_name_value']])]
            if len(c):
                c.to_csv(WORK/'arterial_observations.csv',mode='w' if first else 'a',header=first,index=False);first=False;n+=len(c)
        logging.info('Arterial records retained so far: %d',n)
stats['obs_rows']=n;stats['obs_members']=members
(ROOT/'outputs/local_preparation_summary.json').write_text(json.dumps(stats,indent=2))
logging.info('Input preparation finished')
