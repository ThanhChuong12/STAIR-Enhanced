"""Read-only experiment-artifact analysis; never imports a recommender or torch.

Writes derived tables and a bounded acceptance audit, not new experimental results.
Run from any directory with Python and scipy. Raw artifacts are never modified.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import re
import statistics as st
import subprocess

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/eclipse_multiseed_paper_audit_20261020'
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ('RECALL@10', 'RECALL@20', 'NDCG@10', 'NDCG@20')
DATASETS = ('baby', 'sports', 'electronics')
checks, runs, mechanisms = [], [], []
contracts, evidence_manifests = {}, {}
git_trees, source_cache = {}, {}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(name, ok, detail=''):
    checks.append(dict(check=name, passed=bool(ok), detail=detail))


def save_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def source_digest(commit, name):
    key = (commit, name)
    if key not in source_cache:
        if commit not in git_trees:
            git_trees[commit] = subprocess.check_output(
                ['git', 'ls-tree', '-r', '--name-only', commit], cwd=ROOT
            ).decode('utf-8').splitlines()
        matches = [p for p in git_trees[commit] if p == name or p.endswith('/' + name)]
        if len(matches) != 1:
            raise ValueError((commit, name, matches))
        b = subprocess.check_output(['git', 'show', f'{commit}:{matches[0]}'], cwd=ROOT)
        source_cache[key] = hashlib.sha256(b).hexdigest()
    return source_cache[key]


for arm in ('full', 'wo_ec'):
    for dataset in DATASETS:
        for seed in (1, 2, 3):
            label = f'{arm}/{dataset}/s{seed}'
            if arm == 'full' and seed == 1:
                base = ROOT / 'logs/GD6/STAIR-PEEC'
                if dataset == 'electronics':
                    base /= 'electronics'
                p = base / f'stair_icaart/{dataset}_STAIR_ICAART_pooled_activity_excess_d1.0_seed1'
                log = p.with_name(p.name + '.log')
                bundle = None
            elif arm == 'wo_ec' and seed == 1:
                base = ROOT / 'logs/ICAART/eclipse_wo_ec_seed1_reuse_20261018T152158Z_69603726'
                matches = list(base.rglob(f'{dataset}_eclipse_wo_ec_seed1/manifest.json'))
                if len(matches) != 1:
                    raise ValueError(matches)
                p = matches[0].parent
                log = p / 'training.log'
                bundle = None
            else:
                base = ROOT / f'logs/ICAART/eclipse_{"wo_ec_" if arm == "wo_ec" else ""}seed{seed}'
                p = base / f'artifacts/{dataset}_eclipse_{arm}_seed{seed}'
                log = base / f'logs/{dataset}_eclipse_{arm}_seed{seed}.log'
                bundle = read(base / 'run_manifest.json')
            manifest = read(p / 'manifest.json')
            selected = read(p / 'selected_metrics.json')
            status = read(p / 'run_status.json')
            contract = manifest['checkpoint_contract']
            contracts[(arm, dataset, seed)] = contract
            telemetry = [json.loads(x) for x in (p / 'training_telemetry.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
            events = [json.loads(x) for x in (p / 'evaluation_metrics.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
            check(label + '/completed', status['status'] == 'COMPLETED')
            check(label + '/selected_test_seed_delta', selected['mode'] == 'test'
                  and selected['selected_checkpoint'] is True and selected['seed'] == seed
                  and selected['evidence_delta'] == (1.0 if arm == 'full' else 0.0)
                  and selected['selection'].lower() == 'validation ndcg@20')
            check(label + '/finite_metrics', all(math.isfinite(selected['metrics'][m]) for m in METRICS))
            check(label + '/graph_receipt', selected['graph_fingerprint'] == contract['graph_fingerprint'])
            check(label + '/500_epochs', [x['epoch'] for x in telemetry] == list(range(1, 501)))
            check(label + '/finite_loss', all(math.isfinite(x[k]) for x in telemetry for k in ('loss', 'bpr_loss', 'nlgcl_loss')))
            check(label + '/loss_identity', all(abs(x['loss'] - x['bpr_loss'] - x['weighted_nlgcl_loss']) < 1e-7
                  and abs(x['weighted_nlgcl_loss'] - .01*x['nlgcl_loss']) < 1e-7 for x in telemetry))
            valid = [x for x in events if x['mode'] == 'valid']
            terminal_valid = next((x for x in valid if x['epoch'] == 500
                                   and not x.get('selected_checkpoint')), None)
            if valid:
                best = max(valid, key=lambda x: x['metrics']['NDCG@20'])
                check(label + '/validation_selected_epoch', selected['epoch'] == best['epoch'])
            else:
                check(label + '/legacy_selection_receipt', selected.get('training_reused') is True,
                      'Recovered TEST receipt; original VALID history is in training.log, not native JSONL.')
            selected_events = [x for x in events if x['mode'] == 'test' and x.get('selected_checkpoint')]
            check(label + '/selected_jsonl', any(x['epoch'] == selected['epoch'] and
                  all(abs(x['metrics'][m] - selected['metrics'][m]) < 1e-12 for m in METRICS) for x in selected_events))
            commit = selected.get('training_commit', manifest['commit'])
            for name, digest in contract['sources'].items():
                check(label + '/source/' + name, source_digest(commit, name) == digest)
            if bundle:
                cfg = base / f'configs/{dataset}_eclipse_{arm}_seed{seed}.yaml'
                entry = bundle['configs'][dataset]
                check(label + '/config_sha256', sha(cfg) == entry['sha256'])
            cp = p / 'best_model.pt'
            cp_sha = sha(cp) if cp.is_file() else 'NOT_AVAILABLE_LOCALLY'
            if selected.get('checkpoint_sha256'):
                check(label + '/recovered_checkpoint_sha256', cp_sha == selected['checkpoint_sha256'])
            text = log.read_text(encoding='utf-8', errors='replace')
            fit = re.findall(r'Coach\.fit takes\s+([0-9.eE+-]+)\s+seconds', text)
            check(label + '/single_fit_timer', len(fit) == 1)
            row = dict(arm=arm, dataset=dataset, seed=seed, selected_epoch=selected['epoch'],
                       **{m: selected['metrics'][m] for m in METRICS},
                       preprocessing_seconds=manifest['preprocessing_seconds'],
                       fit_seconds=float(fit[0]) if len(fit) == 1 else None,
                       epoch_train_seconds=sum(x['seconds'] for x in telemetry),
                       peak_training_allocated_mib=max(x['peak_allocated_gib'] for x in telemetry)*1024,
                       peak_training_reserved_mib=max(x['peak_reserved_gib'] for x in telemetry)*1024,
                       first_loss=telemetry[0]['loss'], final_loss=telemetry[-1]['loss'],
                       final_bpr=telemetry[-1]['bpr_loss'], final_raw_lc=telemetry[-1]['nlgcl_loss'],
                       final_weighted_lc=telemetry[-1]['weighted_nlgcl_loss'],
                       validation_selected_ndcg20=best['metrics']['NDCG@20'] if valid else None,
                       validation_terminal_ndcg20=terminal_valid['metrics']['NDCG@20'] if terminal_valid else None,
                       commit=commit, graph_fingerprint=selected['graph_fingerprint'],
                       data_fingerprint=contract['data_fingerprint'],
                       checkpoint_sha256=cp_sha, artifact=str(p.relative_to(ROOT)).replace('\\', '/'),
                       checkpoint_bytes=cp.stat().st_size if cp.is_file() else None,
                       manifest_sha256=sha(p / 'manifest.json'),
                       evaluation_jsonl_sha256=sha(p / 'evaluation_metrics.jsonl'),
                       stdout_log_sha256=sha(log),
                       selected_receipt_sha256=sha(p / 'selected_metrics.json'),
                       telemetry_sha256=sha(p / 'training_telemetry.jsonl'))
            runs.append(row)
            g = manifest['graph'].get('icaart', {})
            evidence_manifests[(arm, dataset, seed)] = g
            if arm == 'full':
                mechanisms.append(dict(dataset=dataset, seed=seed,
                    candidate_pairs=g.get('num_candidate_edges'), positive_pairs=g.get('num_edges_with_positive_evidence'),
                    positive_fraction=g.get('frac_edges_with_positive_evidence'),
                    weighted_coverage=g.get('weighted_evidence_coverage'), r_max=g.get('r_max'),
                    r_positive_mean=g.get('r_positive_mean'), operator_dose=g.get('relative_operator_dose'),
                    polynomial_probe_dose=g.get('relative_polynomial_dose'),
                    zero_score_median_coefficient_ratio=g.get('zero_score_semantic_edge_coefficient_ratio_quantiles', [None]*3)[2],
                    final_nnz=g.get('s_ica_nnz'), users=manifest['graph']['num_users'],
                    items=manifest['graph']['num_items'], train_interactions=g.get('num_raw_train_interactions')))
            elif seed > 1:
                check(label + '/zero_calibration_dose', g['exact_v8_bypass'] is True
                      and g['relative_operator_dose'] == 0 and g['relative_polynomial_dose'] == 0)

for dataset in DATASETS:
    ref = contracts[('full', dataset, 1)]
    for arm in ('full', 'wo_ec'):
        for seed in (1, 2, 3):
            c = contracts[(arm, dataset, seed)]
            for key in ('data_fingerprint', 'features', 'splits', 'candidate_hashes', 'sap_audit'):
                check(f'{arm}/{dataset}/s{seed}/matched_{key}', c[key] == ref[key])
            a, b = dict(c['config']), dict(ref['config'])
            a.pop('seed', None); b.pop('seed', None)
            check(f'{arm}/{dataset}/s{seed}/matched_backbone_config', a == b)
        fp = {r['graph_fingerprint'] for r in runs if r['arm'] == arm and r['dataset'] == dataset}
        check(f'{arm}/{dataset}/seed_invariant_recorded_operator', len(fp) == 1)
    for seed in (2, 3):
        for key in ('raw_semantic_fingerprint', 'counts_hash', 'null_expectation_hash',
                    'assignment_hash', 'evidence_score_hash', 'normalized_cf_fingerprint'):
            a = evidence_manifests[('full', dataset, seed)]
            b = evidence_manifests[('full', dataset, 1)]
            check(f'full/{dataset}/s{seed}/seed_invariant_{key}', a[key] == b[key])
        for key in ('raw_semantic_fingerprint', 'counts_hash', 'null_expectation_hash',
                    'assignment_hash', 'evidence_score_hash', 'normalized_cf_fingerprint'):
            check(f'{dataset}/s{seed}/paired_{key}',
                  evidence_manifests[('full', dataset, seed)][key] == evidence_manifests[('wo_ec', dataset, seed)][key])

canonical = ROOT / 'logs/GD6/STAIR-PEEC/electronics/stair_icaart/electronics_STAIR_ICAART_pooled_activity_excess_d1.0_seed1'
duplicate = ROOT / 'logs/GD6/STAIR-PEEC/stair_icaart/electronics_STAIR_ICAART_pooled_activity_excess_d1.0_seed1'
check('electronics_full_s1_duplicate_metrics', read(canonical/'selected_metrics.json') == read(duplicate/'selected_metrics.json'),
      'Duplicate folder excluded; one training seed, not an additional replication.')

summary, diffs, paired = [], [], []
for dataset in DATASETS:
    for metric in METRICS:
        arrays = {}
        for arm in ('full', 'wo_ec'):
            values = [r[metric] for r in runs if r['dataset'] == dataset and r['arm'] == arm]
            arrays[arm] = values
            summary.append(dict(dataset=dataset, arm=arm, metric=metric, n=3,
                                mean=st.mean(values), sample_sd=st.stdev(values)))
        delta = [a-b for a,b in zip(arrays['full'], arrays['wo_ec'])]
        for seed,d in enumerate(delta,1):
            diffs.append(dict(dataset=dataset, metric=metric, seed=seed,
                              full=arrays['full'][seed-1], wo_ec=arrays['wo_ec'][seed-1], difference=d))
        mean, sd = st.mean(delta), st.stdev(delta)
        se = sd / math.sqrt(3)
        t = mean / se if se else math.inf
        p = float(2*stats.t.sf(abs(t), 2))
        ci = float(stats.t.ppf(.975, 2))*se
        permutations = [abs(st.mean([x*y for x,y in zip(delta, signs)])) for signs in itertools.product((-1,1), repeat=3)]
        signflip = sum(x >= abs(mean)-1e-15 for x in permutations)/len(permutations)
        paired.append(dict(dataset=dataset, metric=metric, n=3, mean_difference=mean,
                           relative_percent=mean/st.mean(arrays['wo_ec'])*100,
                           sd_difference=sd, wins=sum(x>0 for x in delta), losses=sum(x<0 for x in delta),
                           t_df2=t, p_unadjusted=p, ci95_low=mean-ci, ci95_high=mean+ci,
                           exact_signflip_two_sided=signflip, p_holm_primary3=None, p_holm_exploratory12=None))


def holm(rows, dest):
    prev = 0
    for i,row in enumerate(sorted(rows,key=lambda r:r['p_unadjusted'])):
        prev=max(prev,min(1.,(len(rows)-i)*row['p_unadjusted']))
        row[dest]=prev


holm([x for x in paired if x['metric']=='NDCG@20'], 'p_holm_primary3')
holm(paired, 'p_holm_exploratory12')
save_csv('selected_test_runs.csv', runs)
save_csv('mean_std.csv', summary)
save_csv('paired_differences.csv', diffs)
save_csv('paired_statistics.csv', paired)
save_csv('mechanism_diagnostics.csv', mechanisms)
save_csv('artifact_checks.csv', checks)
audit = dict(status='ANALYZED_ARTIFACTS_ONLY', raw_runs=18, checks=len(checks),
             passed=sum(x['passed'] for x in checks), failed=[x for x in checks if not x['passed']],
             scope='JSON/log/config/Git-source/available-checkpoint byte audit; no model evaluation or training rerun.',
             missing_full_seed1_checkpoints=[r['artifact'] for r in runs if r['checkpoint_sha256']=='NOT_AVAILABLE_LOCALLY'],
             hieu_arms='A2/A3 human-reported complete; artifacts not local, metrics/scope not audited.',
             legacy_wo_ec_seed1='Recovered equivalent uncalibrated checkpoints; different source revisions. See prior recovery audit; not identical-source reruns.',
             notes=['Sample SD uses ddof=1; SD itself is not an unbiased estimator.',
                    'n=3 paired t tests are exploratory and assume normal independent seed differences.',
                    'Holm families defined for this retrospective audit, not a preregistration.',
                    'Recorded matching hashes establish artifact correspondence, not absence of leakage or independent metric reproduction.'])
(OUT/'audit_acceptance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
print('NDCG@20 paired summaries:',json.dumps([r for r in paired if r['metric']=='NDCG@20'],indent=2))
