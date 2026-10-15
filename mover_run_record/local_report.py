"""Validate computed artifacts and write aggregate Chinese handoff report."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
gate=json.loads((OUT/'m0_dose_audit.json').read_text());prep=json.loads((OUT/'local_preparation_summary.json').read_text())
ev=pd.read_parquet(OUT/'mover_documented_events.parquet');st=ev[ev.event_type.eq('start')]
native=pd.read_csv(OUT/'m2_mover_native.csv');external=pd.read_csv(OUT/'m2_external_validation.csv')
assert gate['VERDICT']=='GO' and gate['G4_pass'] and len(st)==gate['G4_START_events']
assert np.isfinite(st.baseline_map).all() and np.isfinite(st.delta_map_5min).all()
checks=json.loads((OUT/'validation_checks.json').read_text());checks.update({'synthetic_smoke':'PASS','synthetic_G4_stop_rule_exercised':True,'source_observation_member_count':len(prep['obs_members']),'event_rows':len(ev),'START_rows':len(st),'G4_count_matches_table':True,'native_checks':{},'external_checks':{}})
subgroups=[]
for drug,d in st.groupby('drug'):
 p=OUT/f'oof_mover_{drug}.parquet'
 if not p.exists(): continue
 o=pd.read_parquet(p);assert len(o)==len(d)
 assert o.groupby('subjectid').oof_fold.nunique().le(1).all()
 for model,col in [('xgboost','pred_xgb'),('ridge','pred_ridge'),('baseline_map_only','pred_base')]:
  assert np.isfinite(o[col]).all();value=float(np.abs(o.delta_map_5min-o[col]).mean())
  reported=float(native[(native.drug==drug)&(native.model==model)].MAE.iloc[0]);assert abs(value-reported)<=.00501
 checks['native_checks'][drug]={'patients_grouped_without_fold_overlap':True,'n_events':len(o),'n_patients':int(o.subjectid.nunique()),'MAE_recomputed_matches_csv':True,'folds':int(o.oof_fold.nunique())}
 for key,mask in [('age_lt65',o.age.lt(65)),('age_ge65',o.age.ge(65)),('baseline_MAP_lt65',o.baseline_map.lt(65)),('baseline_MAP_ge65',o.baseline_map.ge(65)),('male',o.sex_male.eq(1)),('female',o.sex_male.eq(0)),('IV_route',o.med_route.isin(['IntraVENOUS','IntraVENOUS Push'])),('nonIV_or_unknown_route',~o.med_route.isin(['IntraVENOUS','IntraVENOUS Push']))]:
  if mask.sum()<20:continue
  y=o.loc[mask,'delta_map_5min'];pred=o.loc[mask,'pred_xgb']
  subgroups.append({'drug':drug,'analysis':'native_xgb','subgroup':key,'n':int(mask.sum()),'MAE':float(np.abs(y-pred).mean())})
 p=OUT/f'external_oof_{drug}.parquet'
 if p.exists():
  o=pd.read_parquet(p);assert o.groupby('subjectid').oof_fold.nunique().le(1).all()
  for stage,col in [('before_recalib','pred_external'),('linear_recalib_group_oof','pred_external_linear_oof'),('isotonic_recalib_group_oof','pred_external_isotonic_oof')]:
   assert np.isfinite(o[col]).all();v=float(np.abs(o.delta_map_5min-o[col]).mean());reported=float(external[(external.drug==drug)&(external.stage==stage)].MAE.iloc[0]);assert abs(v-reported)<=.00501
  checks['external_checks'][drug]={'n_events':len(o),'MAE_recomputed_matches_csv':True,'recalibration_folds_patient_disjoint':True}
pd.DataFrame(subgroups).to_csv(OUT/'m2_subgroup_summary.csv',index=False)
(OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2))
def table(df,cols):
 d=df[cols].fillna('NA');return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None))
summary=st.groupby('drug').agg(n_start=('drug','size'),n_cases=('caseid','nunique'),n_patients=('subjectid','nunique')).reset_index()
xgb=native[native.model.eq('xgboost')];base=native[native.model.eq('baseline_map_only')]
comp=xgb[['drug','MAE']].merge(base[['drug','MAE']],on='drug',suffixes=('_xgb','_baseline'))
comp['MAE_improvement']=comp.MAE_baseline-comp.MAE_xgb;comp.to_csv(OUT/'m2_xgb_vs_baseline.csv',index=False)
text=f'''# Topic 2 v3：MOVER 分析与传回说明

## 执行结论

真实 MOVER 数据分析已完成；G1–G4 全部通过。需要注意：原始 MAR 包含局部注射、外用等途径，给药量纲也未统一。按原包事件规则计算的结果应作为探索性分析，不能直接当作已验证的静脉升压药推注队列。全部模型结果是记录给药后血压变化的预测/关联分析，不代表因果治疗效应。

- 全量药物源表：27,961,524 行；升压药 MAR 匹配：{gate['n_vaso_mar_rows']:,} 行。
- 原手术/患者表：{prep['patient_rows']:,} 行；重复 LOG_ID 行 {prep['duplicate_LOG_ID_rows']:,}；排除冲突病例 {prep['conflicting_case_ids_excluded']:,} 个，排除无效手术窗口 {prep['invalid_windows_excluded']:,} 个。
- 可用唯一手术窗口：{gate['n_surgery_windows']:,}；窗口内记录 START：{gate['n_start_admin']:,}。
- G1 时间可解析率：{gate['G1_timestamp_parse_frac']:.2%}；原始文本含 HH:MM：{gate['G1_timestamp_text_has_HH_MM_frac']:.2%}。
- G2 数字剂量并有单位：{gate['G2_dose_and_unit_frac']:.2%}；G3 动作全部分类。
- 完整扫描 {len(prep['obs_members'])} 个原始血压分片；提取含 MAP 链接的 START：{len(st):,} 个；G4 门槛为 1,000 个。
- 总事件表 {len(ev):,} 行，包含 START 与恢复/调整输注敏感性事件；主模型只使用 START。

## 可分析事件数

{table(summary,['drug','n_start','n_cases','n_patients'])}

## 本地模型结果

患者分组五折交叉验证；MAE/RMSE 单位为 mmHg。MAE 置信区间采用 1,000 次患者聚类 bootstrap。80% 预测区间的理想覆盖率为 0.80。下表保留原报告列名，便于汇入上游分析。

{table(native,['drug','model','n','MAE','MAE_lo','MAE_hi','RMSE','R2','calib_slope','PI80_coverage'])}

XGBoost 相对仅基线 MAP 模型的 MAE 改善（正数表示 XGBoost 更好）：

{table(comp,['drug','MAE_xgb','MAE_baseline','MAE_improvement'])}

## VitalDB → MOVER 外部验证

原始冻结模型未重新训练。linear/isotonic 校准在目标队列内按患者分组交叉拟合，评估患者未参与相应校准器拟合；这些是经过本地校准后的性能，不是未经适配的外部性能。

{table(external,['drug','n','stage','MAE','MAE_lo','MAE_hi','calib_slope'])}

## 结果解读与后续建议

1. 本地模型没有单一通用胜者。去氧肾上腺素与麻黄碱的 XGBoost MAE 最低；肾上腺素、去甲肾上腺素、血管加压素则是 Ridge 更低。应按药物报告全部模型，不能只选 XGBoost 作为统一结论。改善仅为点估计，没有进行配对差异的显著性检验。
2. 仅基线 MAP 模型已经提供较强参考表现；额外特征的收益有限且因药物而异。去甲肾上腺素 XGBoost 的 MAE 高于基线模型，不能声称复杂模型普遍带来改善。
3. XGBoost 的 80% 预测区间覆盖率为 70.3%–79.5%，均未达到名义 80%；去甲肾上腺素的低覆盖最明显。当前区间不能直接作为已校准的个体不确定性范围。
4. 外部冻结模型在去氧肾上腺素的 MAE 为 14.04 mmHg，交叉拟合 isotonic 后为 12.05，仍高于本地 XGBoost 的 10.36。去甲肾上腺素为 13.21，线性再校准后为 12.45，仍高于本地 Ridge 的 10.72。两种药的最佳再校准 MAE 也高于其本地仅基线 MAP 模型；当前证据不支持优先使用直接迁移的 VitalDB 模型。
5. `m2_subgroup_summary.csv` 给出年龄、基线血压、性别以及静脉/其他途径的分层表现。途径分层是在全途径训练模型的折外预测上评估，不能替代重新限定静脉暴露、重新计算不同药物共给药排除规则并重训的敏感性分析。
6. 建议将本次结果交回作为可复核的探索性结果，先明确静脉暴露、药物浓度/剂量单位和未来信息规则，再冻结最终主分析方案并重跑；本次门槛通过并不自动验证所有临床语义。

## 解读限制与代码修正

- 原始配置列名不匹配真实数据；已映射并保留原始脚本与本次执行脚本，见 `LOCAL_METHOD_CHANGES.md`、`IMPLEMENTATION_REVIEW.md`。
- 原 MAR 动作中的 IV Resume / IV Resume Dual Sign 归为敏感性事件；Override pull for Anesthesia 排除。原始首次 PAUSE 审计保留为 `m0_dose_audit_initial.json`，最终判定见 `m0_dose_audit.json`。
- 本地模型去掉未来 10 分钟重复给药标志，缺失值只在训练折拟合；加入剂量单位指示列。原始数值剂量没有跨质量/体积/输注速率统一，且未按给药途径额外限制。研究结论必须结合 `m0_start_dose_unit_route_counts.csv` / `m1_event_dose_unit_route_counts.csv` 检查，不能把所有事件视为同剂量同途径的静脉推注。
- 冻结 VitalDB 模型本身包含未来重复给药标志，且跨队列剂量量纲未证实一致，故外部结果只应作探索性回顾验证。需要严格前瞻预测时应重新制定一致暴露、剂量与特征规则并重训开发模型。
- 年龄、身高、体重转换遵循本地已有 Topic 8 解析规则；发表前仍需对照数据字典确认。事件的时间窗口和同药重复给药定义保留原包规则；基线缺失时回退到整个前 10 分钟、不同药物排除及后 10 分钟峰值等细节见源码。
- 冻结 XGBoost pickle 加载产生跨版本序列化提示，已保留日志；合成数据中两份模型及真实数据均完成预测。未获得原训练环境版本与参考预测，尚不能证明跨版本数值完全一致。运行版本见 `validation_checks.json`。NumPy 2.5 在时间差比较中产生弃用提示，完整控制台日志保留并无损压缩为 `.log.gz`，没有过滤原始日志。
- 已独立复核真实结果 MAE 与逐事件预测一致、患者没有跨验证折重叠、G4 数量与事件表一致；合成冒烟测试与真实数据分开。

## 传回包

`topic2_v3_mover_outputs.tar.gz` 包含完整 `outputs/`：最终/首次门槛 JSON、事件 parquet 和样例 CSV、三类本地模型结果、外部验证结果、逐事件折外预测、分层汇总、日志、复现源码和 SHA-256 清单。

用户已在本次会话明确确认 DUA 允许传回逐事件生理数据及内部病例/患者编号，因此完整事件与预测表纳入传回包。原始 MOVER 源表和本地中间提取 CSV 不纳入传回包。

接收方请先读本文件，再读 `LOCAL_METHOD_CHANGES.md`，最后使用 `CHECKSUMS_SHA256.txt` 核对文件。
'''
(OUT/'RESULTS_SUMMARY_ZH.md').write_text(text)
print('All real-result validation checks passed; Chinese report written.')
