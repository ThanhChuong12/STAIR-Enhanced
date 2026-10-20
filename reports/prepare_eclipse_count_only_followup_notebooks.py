"""Derive Count-only seeds2/3 from the current seed1 notebook, without executing cells."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import copy
import hashlib
import json
import textwrap
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebook/P6/eclipse_count_only_seed1.ipynb'
OUT = ROOT / 'reports/eclipse_count_only_seeds2_3_notebooks'
OUT.mkdir(parents=True, exist_ok=True)
original_bytes = SOURCE.read_bytes()
source_sha = hashlib.sha256(original_bytes).hexdigest()
original = json.loads(original_bytes)


def src(cell):
    return ''.join(cell['source'])


def set_src(cell, source):
    cell['source'] = source.splitlines(keepends=True)


def assignments(source):
    return {n.targets[0].id: n for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}


def make(seed):
    target = ROOT / f'notebook/P6/eclipse_count_only_seed{seed}.ipynb'
    if target.exists():
        raise FileExistsError(f'Preserve existing notebook before rebuilding: {target}')
    nb = copy.deepcopy(original)
    # The seed1 dataset sections occupy cells31..38. Preserve shared engines verbatim.
    assert 'Amazon Electronics' in src(nb['cells'][31])
    assert '## 11.' in src(nb['cells'][39])
    nb['cells'] = nb['cells'][:31] + nb['cells'][39:]
    intro = src(nb['cells'][0])
    intro = intro.replace('training seed 1', f'training seed {seed}')
    intro = intro.replace('**Baby → Sports → Electronics**, three fresh', '**Baby → Sports**, two fresh')
    intro = intro.replace('- Electronics currently requires only seed1. For later seeds2/3 select Baby/Sports only, per plan. No automatic resume or retry.',
                          f'- This notebook runs **Baby/Sports seed{seed} only**. Electronics Count-only is seed1-only under the execution plan. No automatic resume or retry.')
    intro = intro.replace('- Historical T4 timing suggests about10 GPU-hours in total plus setup/export, **not** a guarantee of fitting any platform limit. You may select a dataset subset before starting; do not shorten the epoch budget.',
                          '- Two sequential 500-epoch runs on the selected GPU. You may select Baby or Sports alone before setup; do not shorten the epoch budget. No platform-duration guarantee.')
    set_src(nb['cells'][0], intro)
    config = src(nb['cells'][2])
    assert config.count('SEED = 1\n') == 1
    config = config.replace('SEED = 1\n', f'SEED = {seed}\n')
    config = config.replace("SELECTED_DATASETS = ('baby', 'sports', 'electronics')", "SELECTED_DATASETS = ('baby', 'sports')")
    config = config.replace('default job is Count-only seed1.', f'default job is Count-only seed{seed}.')
    config = config.replace("DATASET_SOURCE_OVERRIDES = {}  # Example: {'electronics': '/kaggle/input/.../electronics'}",
                            "DATASET_SOURCE_OVERRIDES = {}  # Example: {'baby': '/kaggle/input/.../baby'}")
    config = config.replace("if set(SELECTED_DATASETS)-{'baby','sports','electronics'}:\n    raise ValueError('Unknown dataset key.')",
                            "if set(SELECTED_DATASETS)-{'baby','sports'}:\n    raise ValueError('This follow-up notebook supports Baby/Sports only; Electronics Count-only is seed1-only.')")
    set_src(nb['cells'][2], config)
    # Keep frozen seed1 Full protocol references; these are not new seed1 results.
    helpers = src(nb['cells'][4])
    nodes = assignments(helpers)
    reference_node = nodes['SEED1_REFERENCE']
    refs = json.loads(ast.literal_eval(reference_node.value.args[0]))
    references = {k: refs[k] for k in ('baby', 'sports')}
    lines = helpers.splitlines(keepends=True)
    snapshot_node = nodes['NOTEBOOK_SOURCE_TEMPLATE']
    del lines[snapshot_node.lineno-1:snapshot_node.end_lineno]
    lines[reference_node.lineno-1:reference_node.end_lineno] = [
        'SEED1_REFERENCE=json.loads(' + repr(json.dumps(references, separators=(',', ':'))) + ')\n'
    ]
    set_src(nb['cells'][4], ''.join(lines).rstrip()+'\n')
    for cell in nb['cells']:
        cell['id'] = uuid.uuid4().hex[:8]
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
        else:
            s = src(cell)
            s = s.replace('Amazon Baby seed 1', f'Amazon Baby seed {seed}')
            s = s.replace('Amazon Sports seed 1', f'Amazon Sports seed {seed}')
            s = s.replace('## 11. Current-session', '## 10. Current-session')
            s = s.replace('## 12. Complete handoff', '## 11. Complete handoff')
            s = s.replace('Default run order is Baby→Sports→Electronics;', 'Default run order is Baby→Sports;')
            set_src(cell, s)
    nb['metadata']['eclipse'].update(default_seed=seed, datasets=['baby', 'sports'],
        created='2026-10-20', derived_from=SOURCE.relative_to(ROOT).as_posix(),
        derived_from_sha256=source_sha, electronics_scope='seed1_only_not_in_this_notebook')
    # Nonrecursive snapshot must describe this derivative, including the fixed guard.
    snapshot = copy.deepcopy(nb)
    set_src(nb['cells'][4], src(nb['cells'][4]) + '\nNOTEBOOK_SOURCE_TEMPLATE=json.loads(' +
            repr(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'))) + ')\n')

    checks=[]
    def check(name, ok):
        if not ok:
            raise AssertionError((seed, name))
        checks.append(name)
    check('35_cells_no_electronics_section', len(nb['cells']) == 35)
    check('unique_ids', len({c['id'] for c in nb['cells']}) == len(nb['cells']))
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':
            compile(src(cell), f'seed{seed}_cell{i}', 'exec')
            check(f'cell{i}_syntax_and_unexecuted', cell['outputs']==[] and cell['execution_count'] is None)
    setup_nodes=assignments(src(nb['cells'][2]))
    check('training_seed', ast.literal_eval(setup_nodes['SEED'].value)==seed)
    check('two_selected_datasets', ast.literal_eval(setup_nodes['SELECTED_DATASETS'].value)==('baby','sports'))
    check('count_arm', ast.literal_eval(setup_nodes['ARM'].value)=='count_only')
    check('pinned_runtime', ast.literal_eval(setup_nodes['RUNTIME_COMMIT'].value)==original['metadata']['eclipse']['runtime_commit'])
    # Every executable shared engine, input stage, figure and archive cell is unchanged.
    for i in (6,8,10,12,14,16,18,20,22,24,26,28,30):
        check(f'original_cell{i}_unchanged', src(nb['cells'][i])==src(original['cells'][i]))
    for new,old in ((32,40),(34,42)):
        check(f'original_cell{old}_unchanged', src(nb['cells'][new])==src(original['cells'][old]))
    for index, name in ((6,'probe'),(10,'config_probe')):
        nodes=[n for n in ast.walk(ast.parse(src(nb['cells'][index])))
               if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
               and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        check(name+'_present',len(nodes)==1)
        compile(textwrap.dedent(nodes[0].value.value),name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            for node in ast.walk(ast.parse(src(cell))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in (
                    'train_dataset','plot_learning_curves','plot_training_memory','plot_calibration_diagnostics'):
                    check('invocation_'+node.func.id+'_'+str(node.lineno),
                          not node.args or not isinstance(node.args[0],ast.Constant)
                          or node.args[0].value!='electronics')
    helper_nodes=assignments(src(nb['cells'][4]))
    frozen=json.loads(ast.literal_eval(helper_nodes['SEED1_REFERENCE'].value.args[0]))
    check('reference_payload_preserved_baby_sports',frozen==references)
    embedded=json.loads(ast.literal_eval(helper_nodes['NOTEBOOK_SOURCE_TEMPLATE'].value.args[0]))
    check('embedded_snapshot_exact',embedded==snapshot)
    check('snapshot_not_recursive','NOTEBOOK_SOURCE_TEMPLATE' not in assignments(src(embedded['cells'][4])))
    check('fixed_preflight_empty_directory_guard', 'if invalid_directory or has_payload or has_log:' in src(nb['cells'][12])
          and 'artifact.mkdir(parents=True, exist_ok=True)' in src(nb['cells'][12]))
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return dict(notebook=target.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                seed=seed,datasets=['baby','sports'],cells=len(nb['cells']),checks=checks,
                status='CREATED_STATIC_CHECKED_NOT_EXECUTED')


results=[make(seed) for seed in (2,3)]
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=source_sha:
    raise RuntimeError('Seed1 source unexpectedly changed')
receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=SOURCE.relative_to(ROOT).as_posix(),
             source_sha256=source_sha,source_unchanged=True,notebooks=results,
             scope='JSON/AST/compile/diff checks only; no notebook cells, model, GPU, training, network or test suite executed.')
(OUT/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for r in results:
    print(r['notebook'],r['status'],len(r['checks']),'static checks')
