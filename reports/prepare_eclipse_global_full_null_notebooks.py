"""Package the minimal Global-full source overlay and derive three training notebooks.

Only JSON/AST compilation and source comparisons are performed here; no notebook cells run.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import subprocess
import textwrap
import uuid

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'notebook/P6/eclipse_count_only_seed2.ipynb'
OUT = ROOT / 'reports/eclipse_global_full_null_notebooks'
OUT.mkdir(parents=True, exist_ok=True)
BASE = '5212263e87c5c1cfe6e48c91afa90201a48f5929'
FILES = ('stair_icaart_config.py', 'models/stair_icaart_evidence.py', 'models/stair_icaart_graph.py')
overlay = {'base_commit': BASE, 'arm': 'global_full_null', 'files': {}}
for name in FILES:
    baseline = subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=ROOT)
    text = (ROOT/name).read_text(encoding='utf-8')
    compile(text, name, 'exec')
    overlay['files'][name] = dict(base_sha256=hashlib.sha256(baseline).hexdigest(),
        sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), content=text)
overlay_id = hashlib.sha256(json.dumps(overlay,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
(OUT/'source_overlay.json').write_text(json.dumps(overlay,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def source(cell):
    return ''.join(cell['source'])


def put(cell, text):
    cell['source'] = text.splitlines(keepends=True)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(('Expected a unique template segment',before[:150],text.count(before)))
    return text.replace(before,after,1)


def assigns(text):
    return {n.targets[0].id:n for n in ast.parse(text).body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}


original = json.loads(TEMPLATE.read_text(encoding='utf-8'))
prior_path=OUT/'build_receipt.json'
prior={r['notebook']:r['sha256'] for r in json.loads(prior_path.read_text(encoding='utf-8'))['notebooks']} if prior_path.exists() else {}
receipts=[]
for seed in (1,2,3):
    target=ROOT/f'notebook/P6/eclipse_global_full_null_seed{seed}.ipynb'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=prior.get(target.relative_to(ROOT).as_posix()):
        raise FileExistsError(f'Existing notebook preserved: {target}')
    nb=copy.deepcopy(original)
    # Strip the inherited nonrecursive snapshot before any content substitutions.
    helpers=source(nb['cells'][4]);n=assigns(helpers)['NOTEBOOK_SOURCE_TEMPLATE']
    lines=helpers.splitlines(keepends=True);del lines[n.lineno-1:n.end_lineno]
    put(nb['cells'][4],''.join(lines).rstrip()+'\n')
    for cell in nb['cells']:
        s=source(cell).replace('Count-only','Global-full null').replace('count_only','global_full_null')
        s=s.replace('pooled_activity_count','pooled_global_excess')
        s=s.replace('c/(c+4)','[c-mu_global]+/(c+mu_global+4)')
        s=s.replace('positive count score','positive excess score').replace('Positive count-score','Positive excess-score')
        s=s.replace('Count score c / (c + 4)','Excess score r').replace('Count-score quantile','Excess-score quantile')
        s=s.replace('count_scores_may_differ','global_scores_may_differ')
        put(cell,s)
        cell['id']=uuid.uuid4().hex[:8];cell['metadata']={}
        if cell['cell_type']=='code':cell.update(execution_count=None,outputs=[])

    intro=r'''# ECL-global_full_null — ECLIPSE activity-conditioning control | training seed SEED_VALUE

**Baby → Sports**, two fresh 64D/500-epoch runs. Only the null-conditioning rule changes from Full ECLIPSE:

\[
M_{active}=|\{u:h_u>0\}|,\quad \mu^{global}_{ij}=n_i n_j/M_{active},\quad
r^{global}_{ij}=\frac{[c_{ij}-\mu^{global}_{ij}]_+}{c_{ij}+\mu^{global}_{ij}+4},\quad
\widetilde W_{ij}=W_{ij}\exp(r^{global}_{ij}).
\]

Full-user pooled excess, **no user halves and no activity conditioning**. Native mode `pooled_global_excess`, bins `none`, halves0, delta1/tau2; WMSG+CSE eta.1+LC.01+MI/FSC/BSC and per-dataset settings match Full. Inactive TRAIN users do not enter M_active. This is neither Count-only nor `unstratified_replication`.

- Attach processed Baby/Sports inputs before Save & Run All; enable GPU/Internet for setup. RUN_MODE=`train` starts jobs only in the dataset sections. Jobs run sequentially on GPU_INDEX. Electronics is excluded from this plan.
- Source is the recorded Full commit **plus three declared, SHA-locked estimator files embedded here**. A private checkout is used; existing Full/Count checkouts are not patched. No unpublished GitHub commit is required. All other recorded source hashes must still match Full.
- `RUN_MODE='report_only'` accepts only corresponding Global-full receipts; no clone/install/model/CUDA/training in that mode. Never relabel Full/Count receipts as this arm.
- Native VALID NDCG@20 selects the checkpoint; report selected TEST. Keep 500epochs; no TEST-based tuning or automatic retries/resume/deletion.
- PDFs/PNG300dpi, tables/source data/captions and ZIP+checksum are generated after measured runs. A session has one actual seed/dataset, no invented SD/p-values or missing-as-zero values.
'''.replace('SEED_VALUE',str(seed))
    put(nb['cells'][0],intro)
    cfg=source(nb['cells'][2])
    cfg=replace_once(cfg,'SEED = 2\n',f'SEED = {seed}\n')
    cfg=cfg.replace('default job is Global-full null seed2.',f'default job is Global-full null seed{seed}.')
    cfg=cfg.replace('Electronics Global-full null is seed1-only.','Electronics is outside the Global-full plan.')
    cfg=cfg.replace("if SEED != 1 and 'electronics' in SELECTED_DATASETS:","if 'electronics' in SELECTED_DATASETS:")
    cfg=cfg.replace('Plan: Global-full null Electronics seed1 only; select Baby/Sports for seeds2/3.',
                    'Global-full null plan: Baby/Sports only for seeds1/2/3.')
    cfg=cfg.replace("REPO_OVERRIDE = None  # Optional existing repository; checked against seed-1 source digests.",
                    "REPO_OVERRIDE = None  # Disabled for this arm: use an isolated base-plus-overlay checkout.")
    cfg=cfg.replace("/ RUNTIME_COMMIT[:12] / ARM",f"/ RUNTIME_COMMIT[:12] / ARM / '{overlay_id[:12]}'")
    put(nb['cells'][2],cfg)
    helpers=source(nb['cells'][4])
    helpers += '\nGLOBAL_SOURCE_OVERLAY=json.loads('+repr(json.dumps(overlay,ensure_ascii=False,separators=(',',':')))+')\n'
    helpers += f"GLOBAL_OVERLAY_ID={overlay_id!r}\n"
    helpers += '''EXPECTED_RUNTIME_SOURCES=dict(SEED1_REFERENCE['baby']['contract']['sources'])
DECLARED_SOURCE_CHANGES={}
for relative, record in GLOBAL_SOURCE_OVERLAY['files'].items():
    basename=Path(relative).name
    if EXPECTED_RUNTIME_SOURCES[basename]!=record['base_sha256']:
        raise RuntimeError('Overlay base differs from recorded Full reference: '+relative)
    EXPECTED_RUNTIME_SOURCES[basename]=record['sha256']
    DECLARED_SOURCE_CHANGES[relative]={'full_sha256':record['base_sha256'],'global_sha256':record['sha256']}
write_json(RUN_ROOT/'provenance'/'global_full_source_overlay.json',GLOBAL_SOURCE_OVERLAY)
write_json(RUN_ROOT/'provenance'/'declared_source_changes.json',{'base_commit':RUNTIME_COMMIT,
    'overlay_id':GLOBAL_OVERLAY_ID,'changes':DECLARED_SOURCE_CHANGES,
    'scope':'Global-full estimator/config wiring only; Full algorithm sources remain unchanged elsewhere.'})
'''
    put(nb['cells'][4],helpers)
    env=source(nb['cells'][6])
    env=replace_once(env,"REPO_ROOT=Path(REPO_OVERRIDE).expanduser().resolve()",
                     "raise ValueError('REPO_OVERRIDE is disabled: this control uses its own private checkout.')")
    env=replace_once(env,"REPO_ROOT=WORK_DIR/'eclipse_sources'/RUNTIME_COMMIT",
                     "REPO_ROOT=WORK_DIR/'eclipse_sources'/('global_full_null_'+GLOBAL_OVERLAY_ID[:12])/SESSION_ID")
    anchor="    for args,name in ((['status','--porcelain'],'git_status.txt'),"
    insert='''    # Apply only the explicitly declared source extension before any imports/probes.
    overlay_receipt={}
    for relative, record in GLOBAL_SOURCE_OVERLAY['files'].items():
        path=REPO_ROOT/relative
        if path.is_symlink() or not path.is_file():raise ValueError('Invalid overlay target: '+relative)
        current=file_sha256(path)
        current_lf=hashlib.sha256(path.read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest()
        if not ({current,current_lf}&{record['base_sha256'],record['sha256']}):
            raise RuntimeError('Unexpected private-checkout source; refusing overwrite: '+relative)
        payload=record['content'].encode('utf-8')
        if hashlib.sha256(payload).hexdigest()!=record['sha256']:raise RuntimeError('Overlay payload hash mismatch')
        path.write_bytes(payload)
        if file_sha256(path)!=record['sha256']:raise RuntimeError('Overlay write mismatch')
        overlay_receipt[relative]={'base_sha256':record['base_sha256'],'sha256':file_sha256(path)}
    write_json(RUN_ROOT/'provenance'/'applied_global_full_overlay.json',{
        'base_commit':actual,'overlay_id':GLOBAL_OVERLAY_ID,'files':overlay_receipt,'private_checkout':str(REPO_ROOT)})
'''
    if env.count(anchor)!=1:raise ValueError('Missing status capture anchor')
    env=env.replace(anchor,insert+anchor)
    env=env.replace("expected_sources = SEED1_REFERENCE[SELECTED_DATASETS[0]]['contract']['sources']",
                    'expected_sources = EXPECTED_RUNTIME_SOURCES')
    env=env.replace('Algorithm source differs from seed 1:', 'Source differs from declared Full-plus-Global overlay:')
    put(nb['cells'][6],env)
    cfg=source(nb['cells'][10])
    cfg=replace_once(cfg,"config['evidence_mode'] = 'pooled_global_excess'",
                     "config['evidence_mode'] = 'pooled_global_excess'\n        config['evidence_activity_bins'] = 'none'")
    cfg=replace_once(cfg,"expected['evidence_mode'] = 'pooled_global_excess'",
                     "expected['evidence_mode'] = 'pooled_global_excess'\n        expected['evidence_activity_bins'] = 'none'")
    cfg=cfg.replace("'reference_commit': RUNTIME_COMMIT, 'actual_commit': ACTUAL_COMMIT,",
                    "'reference_commit': RUNTIME_COMMIT, 'actual_commit': ACTUAL_COMMIT,\n        'source_overlay_id': GLOBAL_OVERLAY_ID, 'declared_source_changes': DECLARED_SOURCE_CHANGES,")
    put(nb['cells'][10],cfg)
    engine=source(nb['cells'][12])
    engine=engine.replace("dict(ref['icaart_config'],evidence_mode='pooled_global_excess')",
                          "dict(ref['icaart_config'],evidence_mode='pooled_global_excess',evidence_activity_bins='none')")
    engine=replace_once(engine,"for name,expected in ref['evidence_invariants'].items():\n        if e.get(name)!=expected:",
        "for name,expected in ref['evidence_invariants'].items():\n        if name in ('assignment_hash','group_sizes','null_expectation_hash'):continue\n        if e.get(name)!=expected:")
    engine=engine.replace("e.get('null_scope')!='full_activity'","e.get('null_scope')!='global_full'")
    engine=engine.replace("    e=manifest.get('icaart_manifest',{});g=manifest.get('graph',{});d=g.get('icaart',{})",
        """    e=manifest.get('icaart_manifest',{});g=manifest.get('graph',{});d=g.get('icaart',{})
    expected_identity={'null_scope':'global_full','score_kind':'pooled_global_excess',
        'active_partition':False,'active_halves':0,'normalization':'actual_degree'}
    checkpoint_estimator=contract.get('stair_icaart',{})
    for label, snapshot in (('root',manifest),('checkpoint',checkpoint_estimator),
                             ('checkpoint_manifest',checkpoint_estimator.get('manifest',{})),('diagnostics',d)):
        for name,expected in expected_identity.items():
            if snapshot.get(name)!=expected:problems[label+'.'+name]=snapshot.get(name)
    for name in ('num_active_train_users','num_inactive_train_users','num_null_groups','num_activity_strata',
                 'activity_conditioning','evidence_activity_bins','counts_hash','null_expectation_hash',
                 'global_null_formula','global_null_max_abs_formula_error'):
        if d.get(name)!=e.get(name):problems['diagnostic_consistency.'+name]=d.get(name)""")
    lo=engine.index('    # Quantiles do not commute')
    hi=engine.index("    fp=contract.get('graph_fingerprint')",lo)
    engine=engine[:lo]+'''    active_expected=sum(ref['evidence_invariants']['group_sizes'])
    if e.get('num_active_train_users')!=active_expected:problems['M_active']=e.get('num_active_train_users')
    if e.get('num_inactive_train_users')!=e.get('num_users',0)-active_expected:problems['inactive_users']=e.get('num_inactive_train_users')
    if e.get('group_sizes')!=[active_expected]:problems['global_pool']=e.get('group_sizes')
    if e.get('num_null_groups')!=1 or e.get('num_activity_strata')!=0 or e.get('activity_conditioning') is not False:
        problems['activity_conditioning']='Global-full must contain one non-stratified active-user pool'
    if e.get('evidence_activity_bins')!='none':problems['activity_bins']=e.get('evidence_activity_bins')
    if e.get('global_null_formula')!='n_i*n_j/M_active; binary TRAIN item marginals':problems['null_formula']=e.get('global_null_formula')
    error=e.get('global_null_max_abs_formula_error')
    if isinstance(error,bool) or not isinstance(error,(int,float)) or not math.isfinite(error) or error<0:
        problems['global_null_formula_check']='missing/invalid candidate-wise formula check'
    for name in ('null_expectation_hash','assignment_hash'):
        if not isinstance(e.get(name),str) or re.fullmatch(r'[0-9a-f]{64}',e[name]) is None:problems[name]=e.get(name)
    sq=d.get('evidence_score_quantiles')
    if not sq or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<1 for v in sq):
        problems['global_excess_score_range']=sq
''' +engine[hi:]
    engine=engine.replace("for k,v in ref['contract']['sources'].items() if contract.get('sources',{}).get(k)!=v}",
                          "for k,v in EXPECTED_RUNTIME_SOURCES.items() if contract.get('sources',{}).get(k)!=v}")
    engine=engine.replace("'model_intervention':{'evidence_mode':['pooled_activity_excess','pooled_global_excess']}",
        "'model_intervention':{'null_scope':['full_activity','global_full'],'evidence_mode':['pooled_activity_excess','pooled_global_excess']},'declared_source_changes':DECLARED_SOURCE_CHANGES,'source_overlay_id':GLOBAL_OVERLAY_ID")
    engine=engine.replace("'same_candidates_raw_weights_counts_null':True", "'same_candidates_raw_weights_counts_cf':True,'same_activity_null':'NOT_REQUIRED'")
    engine=engine.replace("'purpose':'component_ablation'","'purpose':'evidence_control_global_full_null'")
    engine=engine.replace("problems['count_score_hash']", "problems['global_score_hash']")
    # Export the new scalar diagnostics alongside existing summary data.
    engine=engine.replace("    RESULTS[key] = summary", """    write_json(RUN_ROOT/'tables'/f'{stem}_global_null_audit.json',
        {name:manifest['icaart_manifest'].get(name) for name in ('null_scope','activity_conditioning',
          'num_active_train_users','num_inactive_train_users','num_null_groups','num_activity_strata',
          'global_null_formula','global_null_max_abs_formula_error','counts_hash','null_expectation_hash')})
    RESULTS[key] = summary""")
    put(nb['cells'][12],engine)
    plots=source(nb['cells'][14])
    plots=plots.replace('Global-full null uses [c-mu_global]+/(c+mu_global+4) without subtracting the null;',
                        'Global-full null uses positive excess over n_i*n_j/M_active with denominator c+mu_global+4;')
    plots=plots.replace("'probe_scope':diagnostics.get('polynomial_probe_scope')}",
                        "'probe_scope':diagnostics.get('polynomial_probe_scope'),'null_scope':'global_full','num_active_train_users':diagnostics.get('num_active_train_users')}")
    put(nb['cells'][14],plots)
    archive=source(nb['cells'][34])
    archive=archive.replace('Count score: [c-mu_global]+/(c+mu_global+4), no null subtraction; null may still be computed offline.',
        'Global excess score: [c-mu_global]+/(c+mu_global+4), mu_global=n_i*n_j/M_active; no user halves/activity strata.')
    archive=archive.replace('Pinned algorithm source: {RUNTIME_COMMIT}. Source template excludes recursive self-embedding.',
        'Algorithm source: base {RUNTIME_COMMIT} + declared Global-full overlay {GLOBAL_OVERLAY_ID}. See source digests and diff. Source template excludes recursive self-embedding.')
    put(nb['cells'][34],archive)
    # Correct residual explanatory labels; executable mode/seed checks remain explicit.
    for cell in nb['cells']:
        s=source(cell)
        if cell['cell_type']=='markdown':
            s=s.replace('seed 2',f'seed {seed}')
            if s.startswith('## 2.'):
                s='''## 2. Frozen Full protocol and declared Global-full extension
Input/candidate/count/raw semantic/CF hashes remain fixed. Null expectation, one-pool assignment, scores and calibrated final graph may differ. Full source references remain immutable; only three declared estimator files have new source digests.
'''
            if s.startswith('## 3.'):
                s='''## 3. Private pinned source + declared overlay and matched runtime
Clone the recorded Full revision into a private session directory, apply the embedded three-file extension after base-hash checks, and save source hashes/full diff. No existing training checkout is modified. No fallback HEAD, anonymous monkey-patch or Torch upgrade. All unmodified source digests must match Full.
'''
            if s.startswith('## 5.'):
                s='''## 5. Resolved configuration and native CLI preflight
Mode=pooled_global_excess, bins=none, halves0, delta1/tau2; eta.1/LC.01 and per-dataset training hyperparameters remain Full. Native resolved settings are checked before training.
'''
            if s.startswith('## 6.'):
                s='''## 6. Global-full guards and output engine
Checks selected TEST/seed/config/status, 500epochs, objective identity, source extension, invariant candidates/counts/CF, one active-user pool, null formula audit and score range. Null/assignment/score/final operator are allowed to differ from activity-conditioned Full.
'''
            if s.startswith('## 7.'):
                s='''## 7. Figure utilities — PDF vector / PNG300dpi / measured data / captions
Training objective, VALID ranking, training allocator/time and Global-full excess-score diagnostics. No TEST learning-curve selection or phantom error bars. Positive coverage means c exceeds the global active-user null, not merely c>0. Missing diagnostics remain NOT_MEASURED.
'''
        put(cell,s)
    nb['metadata']['eclipse'].update(arm='global_full_null',default_seed=seed,datasets=['baby','sports'],
        created='2026-10-20',derived_from=TEMPLATE.relative_to(ROOT).as_posix(),
        derived_from_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        source_overlay_id=overlay_id,source_base_commit=BASE,electronics_scope='not_in_plan')
    snapshot=copy.deepcopy(nb)
    put(nb['cells'][4],source(nb['cells'][4])+'\nNOTEBOOK_SOURCE_TEMPLATE=json.loads('+
        repr(json.dumps(snapshot,ensure_ascii=False,separators=(',',':')))+')\n')
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError((seed,name))
        checks.append(name)
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(source(cell),f'global_s{seed}_cell{i}','exec')
            check(f'cell{i}_syntax_unexecuted',cell['outputs']==[] and cell['execution_count'] is None)
    for index,name in ((6,'probe'),(10,'config_probe')):
        node=next(n for n in ast.walk(ast.parse(source(nb['cells'][index])))
                  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
                  and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        compile(textwrap.dedent(node.value.value),name,'exec');checks.append(name+'_syntax')
    setup=assigns(source(nb['cells'][2]));h=assigns(source(nb['cells'][4]))
    check('seed_default',ast.literal_eval(setup['SEED'].value)==seed)
    check('baby_sports_only',ast.literal_eval(setup['SELECTED_DATASETS'].value)==('baby','sports'))
    check('arm_global_full',ast.literal_eval(setup['ARM'].value)=='global_full_null')
    check('embedded_overlay_hash',ast.literal_eval(h['GLOBAL_OVERLAY_ID'].value)==overlay_id)
    embedded=json.loads(ast.literal_eval(h['NOTEBOOK_SOURCE_TEMPLATE'].value.args[0]))
    check('snapshot_correct_and_nonrecursive',embedded==snapshot and 'NOTEBOOK_SOURCE_TEMPLATE' not in assigns(source(embedded['cells'][4])))
    check('same_training_and_input_engine',source(nb['cells'][8])==source(original['cells'][8]))
    check('global_null_guard',"e.get('null_scope')!='global_full'" in engine)
    check('no_incorrect_count_extrema_formula','expected=cvalue/(cvalue+4.0)' not in engine)
    check('declared_source_validation', 'for k,v in EXPECTED_RUNTIME_SOURCES.items()' in engine)
    check('private_checkout',"('global_full_null_'+GLOBAL_OVERLAY_ID[:12])/SESSION_ID" in env)
    check('fixed_empty_preflight_guard','if invalid_directory or has_payload or has_log:' in engine)
    check('global_bins_config_probe',"expected['evidence_activity_bins'] = 'none'" in source(nb['cells'][10]))
    executable='\n'.join(source(c) for i,c in enumerate(nb['cells']) if c['cell_type']=='code' and i!=4)
    check('no_count_only_mode', 'pooled_activity_count' not in executable)
    check('no_electronics_training_call',"train_dataset('electronics')" not in executable)
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    receipts.append(dict(notebook=target.relative_to(ROOT).as_posix(),seed=seed,datasets=['baby','sports'],
        cells=len(nb['cells']),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),checks=checks,
        status='READY_STATIC_CHECKED_NOT_EXECUTED'))

(OUT/'build_receipt.json').write_text(json.dumps(dict(base_commit=BASE,source_overlay_id=overlay_id,
    template=TEMPLATE.relative_to(ROOT).as_posix(),template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
    notebooks=receipts,scope='AST/compile/hash checks only; notebook cells, model/GPU/training/network not executed.'),
    ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in receipts:print(r['notebook'],len(r['checks']),'static checks')
