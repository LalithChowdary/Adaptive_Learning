"""Assemble measured release inventory after all audits pass; no model training."""
import json
from pathlib import Path
from catalog import FORMATS, SKILLS
from simulator import VERSION
from storage import digest, save_json
from replay import COMMON, FIELDS, FEATURE_FIELDS, TASK_FIELDS


def main():
    root=Path(__file__).parent; dataset=root/'dataset'
    cohorts={name:json.loads((dataset/name/'audit.json').read_text()) for name in ('pilot','main','power_law','long_gap')}
    learners=set(); counts={}; totals=dict(learners=0,events=0,responses=0,hints=0,retries=0,tasks=0)
    for name,report in cohorts.items():
        if report['status']!='passed': raise ValueError('failed audit')
        folder=dataset/name
        if digest(folder/'manifest.json')!=report['audited_manifest_sha256']: raise ValueError('stale audit')
        manifest=json.loads((folder/'manifest.json').read_text())
        for relative,checksum in manifest['source_files'].items():
            if digest(root/relative)!=checksum: raise ValueError('source changed since generation')
        for relative,checksum in manifest['files'].items():
            if digest(folder/relative)!=checksum: raise ValueError('data changed since audit')
        for relative,checksum in manifest['catalog_files'].items():
            if digest(dataset/relative)!=checksum: raise ValueError('catalog changed')
        config=json.loads((folder/'config.json').read_text())
        ids={a[0] for a in config['assignments']}
        if ids & learners: raise ValueError('cohorts overlap')
        learners |= ids
        counts[name]=report['splits']
        for stats in report['splits'].values():
            for key,source in (('learners','learners'),('events','events'),('responses','event_response'),('hints','event_hint'),('retries','event_retry'),('tasks','event_task')):
                totals[key]+=stats[source]
    tests=json.loads((root/'reports/test_results.json').read_text())
    if not tests['successful']: raise ValueError('tests did not pass')
    preserved=json.loads((root/'reports/v1_preservation.json').read_text())
    if any(digest(root.parent/path)!=checksum for path,checksum in preserved.items()): raise ValueError('v1 was changed')
    summary=dict(version=VERSION,totals=totals,cohorts=counts,tests={k:tests[k] for k in ('tests_run','passed','failed','errors','skipped')},
                 audit_categories={name:r['check_categories_passed'] for name,r in cohorts.items()},v1_preserved_files=len(preserved),
                 main_training_targets=cohorts['main']['splits']['train']['eligible_practice'],
                 missing_assessment_coverage={name:r['missing_assessment_coverage_cells'] for name,r in cohorts.items()},
                 cohort_manifest_hashes={name:digest(dataset/name/'manifest.json') for name in cohorts},
                 model_trained=False)
    save_json(root/'reports/RELEASE_SUMMARY.json',summary)
    save_json(dataset/'schema_fields.json',dict(version=VERSION,raw_common=sorted(COMMON),raw_by_event={k:sorted(v) for k,v in FIELDS.items()},memory_event=sorted(FEATURE_FIELDS),memory_task=sorted(TASK_FIELDS)))
    files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='RELEASE_MANIFEST.json'}
    save_json(root/'RELEASE_MANIFEST.json',dict(version=VERSION,files=files))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
