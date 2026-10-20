"""Small CPU mathematical/compatibility checks, no recommender/model/test-suite execution."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types

import numpy as np
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/eclipse_global_full_null_notebooks'
checks=[]


def check(name, ok):
    if not ok:raise AssertionError(name)
    checks.append(name)


def rejects(name, fn):
    try:fn()
    except (ValueError,TypeError):checks.append(name);return
    raise AssertionError('Expected rejection: '+name)


def load_standalone(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cfg=load_standalone('global_config_validation',ROOT/'stair_icaart_config.py')
ev=load_standalone('global_evidence_validation',ROOT/'models/stair_icaart_evidence.py')
base_cfg=types.ModuleType('base_config')
base_ev=types.ModuleType('base_evidence')
base='5212263e87c5c1cfe6e48c91afa90201a48f5929'
for module,name in ((base_cfg,'stair_icaart_config.py'),(base_ev,'models/stair_icaart_evidence.py')):
    source=subprocess.check_output(['git','show',f'{base}:{name}'],cwd=ROOT).decode('utf-8')
    exec(compile(source,'recorded_'+name,'exec'),module.__dict__)

# Inactive-user exclusion is observable here: M_active=4 while catalog user count=6.
R=sp.csr_matrix(np.array([[1,1,1,0,0],[1,1,0,0,0],[1,0,0,0,0],
                          [0,1,0,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.int8))
pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[3,4]],dtype=np.int64)
global_bins=np.array([0,0,0,0,-1,-1],dtype=np.int64)
sizes=np.array([4],dtype=np.int64)
c,mu=ev.candidate_pooled_statistics(R.tocsc(),pairs,global_bins,sizes,candidate_block=2,null_scope='global_full')
check('observed_counts_hand_calculated',np.array_equal(c,np.array([2,1,0,1,1,0])))
check('active_user_global_null_hand_calculated',np.array_equal(mu,np.array([2.25,.75,.75,.75,.75,0.])))
r=ev.pooled_activity_score(c,mu,2.,'pooled_global_excess')
check('excess_not_count_boost',r[0]==0 and c[0]>0)
check('pooled_regularizer_four',np.allclose(r,np.array([0,.25/5.75,0,.25/5.75,.25/5.75,0]),rtol=0,atol=1e-15))
check('empty_item_null_and_score_zero',mu[-1]==0 and r[-1]==0)
for block in (1,3,8192):
    ca,ma=ev.candidate_pooled_statistics(R.tocsc(),pairs,global_bins,sizes,block,null_scope='global_full')
    check(f'block_invariance_{block}',np.array_equal(c,ca) and np.array_equal(mu,ma))
bins,bs=ev.build_activity_strata(R)
ca,ma=ev.candidate_pooled_statistics(R.tocsc(),pairs,bins,bs)
cb,mb=base_ev.candidate_pooled_statistics(R.tocsc(),pairs,bins,bs)
check('legacy_activity_statistics_bitwise_preserved',np.array_equal(ca,cb) and np.array_equal(ma,mb))
check('only_null_changes_counts_preserved',np.array_equal(c,ca) and not np.array_equal(mu,ma))
rejects('global_rejects_activity_groups',lambda:ev.candidate_pooled_statistics(R.tocsc(),pairs,bins,bs,null_scope='global_full'))
rejects('activity_rejects_global_groups',lambda:ev.candidate_pooled_statistics(R.tocsc(),pairs,global_bins,sizes))
rejects('unknown_null_scope',lambda:ev.candidate_pooled_statistics(R.tocsc(),pairs,global_bins,sizes,null_scope='unknown'))

# Explicit stored duplicates must reduce to TRAIN presence before marginal counts.
coo=R.tocoo(); rows=np.r_[coo.row, np.zeros(200,dtype=np.int64)]
cols=np.r_[coo.col,np.zeros(200,dtype=np.int64)]
duplicate=sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=R.shape).tocsc()
cd,md=ev.candidate_pooled_statistics(duplicate,pairs,global_bins,sizes,null_scope='global_full')
check('duplicate_presence_not_frequency',np.array_equal(cd,c) and np.array_equal(md,mu))
high=sp.csc_matrix((np.ones(420,dtype=np.int8),(np.tile(np.arange(210),2),np.repeat([0,1],210))),shape=(212,3))
hc,hm=ev.candidate_pooled_statistics(high,np.array([[0,1]]),np.r_[np.zeros(210,dtype=np.int64),[-1,-1]],np.array([210]),null_scope='global_full')
check('counts_beyond_int8_no_overflow',hc[0]==210 and np.isclose(hm[0],210,rtol=0,atol=1e-12))
empty=sp.csc_matrix((3,2))
ec,em=ev.candidate_pooled_statistics(empty,np.array([[0,1]]),np.array([-1,-1,-1]),np.array([0]),null_scope='global_full')
check('zero_active_users_safe',ec[0]==0 and em[0]==0)

global_settings=cfg.validate_icaart_config(dict(evidence_mode='pooled_global_excess',evidence_activity_bins='none',evidence_halves=0))
identity=cfg.evidence_identity(global_settings)
check('global_identity_zero_halves',identity['null_scope']=='global_full' and identity['active_halves']==0 and identity['active_partition'] is False and identity['pooled_regularizer']==4.)
check('inactive_partition_seed_none',global_settings['evidence_seed'] is None)
rejects('global_config_rejects_log2_label',lambda:cfg.validate_icaart_config(dict(evidence_mode='pooled_global_excess',evidence_halves=0)))
rejects('global_config_rejects_halves2',lambda:cfg.validate_icaart_config(dict(evidence_mode='pooled_global_excess',evidence_activity_bins='none',evidence_halves=2)))
for mode in base_cfg.ALLOWED_EVIDENCE_MODES:
    values=dict(evidence_mode=mode,evidence_halves=0 if mode in base_cfg.FULL_ACTIVITY_MODES else 2)
    old=base_cfg.validate_icaart_config(values);new=cfg.validate_icaart_config(values)
    check('legacy_config_and_identity_'+mode,old==new and base_cfg.evidence_identity(old)==cfg.evidence_identity(new))

build=json.loads((OUT/'build_receipt.json').read_text(encoding='utf-8'))
overlay=json.loads((OUT/'source_overlay.json').read_text(encoding='utf-8'))
for path,record in overlay['files'].items():
    check('overlay_matches_source_'+path,hashlib.sha256((ROOT/path).read_text(encoding='utf-8').encode()).hexdigest()==record['sha256'])
for rec in build['notebooks']:
    path=ROOT/rec['notebook'];nb=json.loads(path.read_text(encoding='utf-8'))
    check(path.stem+'_build_hash',hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256'])
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),path.stem,'exec')
    # Guards consume native manifests; a minimal synthetic manifest tests only acceptance,
    # never fabricates ranking metrics or writes run artifacts.
    helper_ast=ast.parse(''.join(nb['cells'][4]['source']))
    assign={n.targets[0].id:n for n in helper_ast.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    refs=json.loads(ast.literal_eval(assign['SEED1_REFERENCE'].value.args[0]))
    expected=dict(refs['baby']['contract']['sources'])
    expected.update({Path(p).name:r['sha256'] for p,r in overlay['files'].items()})
    import copy,math,re
    for dataset in ('baby','sports'):
        ref=refs[dataset]
        native=json.loads((ROOT/ref['provenance']/'manifest.json').read_text(encoding='utf-8'))
        m=copy.deepcopy(native);seed=rec['seed'];m['seed']=seed
        m['icaart_config'].update(evidence_mode='pooled_global_excess',evidence_activity_bins='none')
        m['checkpoint_contract']['config']['seed']=seed
        m['checkpoint_contract']['sources']=expected
        e=m['icaart_manifest'];active=sum(ref['evidence_invariants']['group_sizes'])
        e.update(evidence_mode='pooled_global_excess',score_kind='pooled_global_excess',null_scope='global_full',
                 evidence_activity_bins='none',training_seed=seed,num_active_train_users=active,
                 num_inactive_train_users=e['num_users']-active,group_sizes=[active],num_null_groups=1,
                 num_activity_strata=0,activity_conditioning=False,
                 global_null_formula='n_i*n_j/M_active; binary TRAIN item marginals',global_null_max_abs_formula_error=0.)
        identity_keys=('null_scope','score_kind','active_partition','active_halves','normalization')
        m.update({k:e[k] for k in identity_keys})
        m['graph']['icaart'].update(e)
        m['checkpoint_contract']['stair_icaart'].update({k:e[k] for k in identity_keys})
        m['checkpoint_contract']['stair_icaart']['manifest']=copy.deepcopy(e)
        fn=next(n for n in ast.parse(''.join(nb['cells'][12]['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='compare_manifest')
        ns=dict(SEED1_REFERENCE=refs,EXPECTED_RUNTIME_SOURCES=expected,DECLARED_SOURCE_CHANGES={},
                GLOBAL_OVERLAY_ID=build['source_overlay_id'],ARM='global_full_null',RUN_MODE='train',STRICT_RUNTIME=True,math=math,re=re)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'manifest_guard_only','exec'),ns)
        verdict=ns['compare_manifest'](dataset,m,seed)
        check(f'{path.stem}_{dataset}_global_guard_accepts_declared_contract',verdict['status']=='MATCH')
        wrong=copy.deepcopy(m);wrong['icaart_manifest']['null_scope']='full_activity'
        try:ns['compare_manifest'](dataset,wrong,seed)
        except RuntimeError:checks.append(f'{path.stem}_{dataset}_guard_rejects_wrong_null')
        else:raise AssertionError('Guard accepted wrong null')

result=dict(status='CPU_MATH_AND_STATIC_VALIDATED_NOT_TRAINED',checks=checks,count=len(checks),
    build_static_checks=sum(len(r['checks']) for r in build['notebooks']),
    scope='Small synthetic sparse TRAIN examples and standalone config/evidence/manifest guard only; no model, GPU, notebook execution, network or pytest suite.',
    caveat='No Kaggle integration or real-dataset training run has been executed.')
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(result['status'],len(checks),'checks; no training/model execution.')
