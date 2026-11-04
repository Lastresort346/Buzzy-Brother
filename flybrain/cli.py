# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
"""Versioned source acquisition, SQLite inspection, reversible subgraphs, CPU benchmark."""
import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = DATA / 'raw' / 'flywire-783'
DB = DATA / 'connectome.sqlite'
ZENODO = 'https://zenodo.org/records/10676866'
ANNOTATIONS = 'https://github.com/flyconnectome/flywire_annotations'
NT = ['gaba', 'ach', 'glut', 'oct', 'ser', 'da']
SOURCES = {
    'proofread_connections_783.feather': 'f48f972d262323a102aed49af1396b8a',
    'proofread_root_ids_783.npy': 'e0e6c19732fd8c7a4e39a2d170105421',
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def fetch(url, destination, expected_md5=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_md5 and digest(destination, 'md5') != expected_md5:
            raise ValueError(f'Existing source checksum mismatch: {destination}. Source was not overwritten.')
        return
    partial = destination.with_name(destination.name + '.part')
    req = Request(url, headers={'User-Agent': 'FlyBrain-research/0.1'})
    print(f'Downloading {destination.name}', flush=True)
    with urlopen(req, timeout=90) as response, partial.open('wb') as out:
        total = 0
        last = time.monotonic()
        for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
            out.write(block)
            total += len(block)
            if time.monotonic() - last > 10:
                print(f'  {total / 1e6:.1f} MB received', flush=True)
                last = time.monotonic()
    if expected_md5 and digest(partial, 'md5') != expected_md5:
        raise ValueError(f'Checksum mismatch for {partial}; incomplete source was not promoted.')
    partial.rename(destination)


def acquire(args):
    requested_commit = args.annotation_commit
    if requested_commit and (len(requested_commit) != 40 or
                             any(c not in '0123456789abcdef' for c in requested_commit)):
        raise ValueError('--annotation-commit must be a full lowercase 40-character Git SHA')
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if requested_commit and requested_commit != manifest['annotation_commit']:
            raise ValueError('Existing sources use a different annotation revision; originals were not overwritten.')
        for name, entry in manifest['files'].items():
            if not (RAW / name).exists():
                fetch(entry['url'], RAW / name, entry.get('md5'))
        verify(RAW)
        print('Existing source manifest verified; nothing changed.')
        return
    # Pin the annotation source before fetching so later upstream edits cannot change this acquisition.
    lock_path = RAW / 'acquisition-lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        if requested_commit and requested_commit != lock['annotation_commit']:
            raise ValueError('Requested commit differs from existing acquisition lock.')
    else:
        commit = requested_commit
        if commit is None:
            url = 'https://api.github.com/repos/flyconnectome/flywire_annotations/commits/main'
            with urlopen(Request(url, headers={'User-Agent': 'FlyBrain-research/0.1'}), timeout=30) as r:
                commit = json.load(r)['sha']
        lock = {'annotation_commit': commit, 'created_utc': stamp()}
        write_json(lock_path, lock)
    commit = lock['annotation_commit']
    files = {}
    for name, md5 in SOURCES.items():
        url = f'{ZENODO}/files/{name}?download=1'
        fetch(url, RAW / name, md5)
        files[name] = {'url': url, 'md5': md5}
    name = 'Supplemental_file1_neuron_annotations.tsv'
    url = f'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{commit}/supplemental_files/{name}'
    fetch(url, RAW / name)
    files[name] = {'url': url, 'commit': commit}
    for name, entry in files.items():
        entry.update(sha256=digest(RAW / name), bytes=(RAW / name).stat().st_size)
    manifest = {'dataset': 'FlyWire FAFB', 'connectivity_version': '783.0',
                'created_utc': stamp(), 'annotation_commit': commit,
                'connectivity_source': ZENODO, 'annotation_source': ANNOTATIONS,
                'files': files}
    write_json(manifest_path, manifest)
    print('Source files preserved and manifest recorded.')


def verify(raw):
    manifest = json.loads((raw / 'manifest.json').read_text(encoding='utf-8'))
    for name, entry in manifest['files'].items():
        if digest(raw / name) != entry['sha256']:
            raise ValueError(f'Source SHA-256 mismatch: {name}')
    return manifest


def source_fingerprint(manifest):
    """Stable across independent downloads of identical source bytes."""
    identity = {'dataset': manifest['dataset'],
                'connectivity_version': manifest.get('connectivity_version'),
                'annotation_commit': manifest['annotation_commit'],
                'files': {name: entry['sha256'] for name, entry in manifest['files'].items()}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@contextmanager
def connect(path=DB):
    if not Path(path).exists():
        raise ValueError('No database yet. Run: python -m flybrain build')
    conn = sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True)
    try:
        yield conn
    finally:
        conn.close()


def build_database(raw, target):
    import numpy as np
    import pyarrow as pa
    manifest = verify(raw)
    if target.exists():
        raise ValueError(f'Database already exists: {target}. Choose another --output to rebuild.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.building')
    if temp.exists():
        raise ValueError(f'Previous incomplete build exists: {temp}; choose another --output.')
    conn = sqlite3.connect(temp)
    try:
        conn.executescript('''
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=-65536;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE neurons(root_id INTEGER PRIMARY KEY, annotations TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE edges(pre INTEGER NOT NULL, post INTEGER NOT NULL,
                neuropil TEXT NOT NULL, syn_count INTEGER NOT NULL CHECK(syn_count > 0),
                nt_prediction TEXT NOT NULL, nt_confidence REAL);
        ''')
        ids = np.load(raw / 'proofread_root_ids_783.npy', allow_pickle=False)
        conn.executemany('INSERT INTO neurons(root_id) VALUES (?)', ((int(x),) for x in ids))
        with (raw / 'Supplemental_file1_neuron_annotations.tsv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            fields = reader.fieldnames
            if 'root_id' not in fields:
                raise ValueError(f'Unexpected annotation schema: {fields}')
            matched = 0
            unknown = 0
            for row in reader:
                root_id = int(row.pop('root_id'))
                row = {k: v for k, v in row.items() if v and v.lower() not in ('nan', 'na', 'none')}
                changed = conn.execute('UPDATE neurons SET annotations=? WHERE root_id=?',
                                       (json.dumps(row, ensure_ascii=False), root_id)).rowcount
                matched += changed
                unknown += not changed
        conn.execute('INSERT INTO metadata VALUES (?,?)', ('annotation_import', json.dumps(
            {'matched': matched, 'outside_proofread_ids': unknown, 'fields': fields})))
        # Feather v2 is Arrow IPC: read record batches, never expand the full graph into Python objects.
        with pa.memory_map(str(raw / 'proofread_connections_783.feather'), 'r') as source:
            reader = pa.ipc.open_file(source)
            required = ['pre_pt_root_id', 'post_pt_root_id', 'neuropil', 'syn_count']
            for field in required + [x + '_avg' for x in NT]:
                if field not in reader.schema.names:
                    raise ValueError(f'Missing source column: {field}')
            rows = 0
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                columns = [batch.column(batch.schema.get_field_index(x)).to_pylist() for x in required]
                probs = np.column_stack([batch.column(batch.schema.get_field_index(x + '_avg')).to_numpy(
                    zero_copy_only=False) for x in NT])
                safe = np.where(np.isfinite(probs), probs, -1)
                nt_index = np.argmax(safe, axis=1)
                confidence = safe[np.arange(len(safe)), nt_index]
                conn.executemany('INSERT INTO edges VALUES (?,?,?,?,?,?)', (
                    (int(pre), int(post), neuropil or 'UNKNOWN', int(count),
                     NT[int(nt_index[j])] if confidence[j] >= 0 else 'unknown',
                     float(confidence[j]) if confidence[j] >= 0 else None)
                    for j, (pre, post, neuropil, count) in enumerate(zip(*columns))))
                rows += batch.num_rows
                if i % 25 == 0:
                    print(f'Imported {rows:,} connection-region rows', flush=True)
        print('Building graph indexes...', flush=True)
        conn.executescript('CREATE INDEX edges_pre ON edges(pre); CREATE INDEX edges_post ON edges(post);')
        dangling = conn.execute('''SELECT COUNT(*) FROM edges e LEFT JOIN neurons a ON a.root_id=e.pre
            LEFT JOIN neurons b ON b.root_id=e.post WHERE a.root_id IS NULL OR b.root_id IS NULL''').fetchone()[0]
        if dangling:
            raise ValueError(f'{dangling} edges reference IDs outside the proofread neuron list')
        conn.execute('INSERT INTO metadata VALUES (?,?)', ('source_manifest', json.dumps(manifest)))
        conn.execute('INSERT INTO metadata VALUES (?,?)', ('built_utc', stamp()))
        conn.commit()
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Database integrity check failed')
    finally:
        conn.close()
    temp.rename(target)
    print(f'Database ready: {target}')


def search(args):
    with connect(args.database) as conn:
        # Literal substring search. No SQL fragments accepted from CLI.
        result = conn.execute('SELECT root_id,annotations FROM neurons WHERE instr(lower(annotations),lower(?))>0 ORDER BY root_id LIMIT ?',
                              (args.text, args.limit))
        print(json.dumps([{'root_id': str(i), 'annotations': json.loads(a)} for i, a in result], indent=2))


def inspect_database(database, output):
    with connect(database) as conn:
        n = conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]
        edges, synapses = conn.execute('SELECT COUNT(*),SUM(syn_count) FROM edges').fetchone()
        print('Counting directed neuron pairs with a sequential graph scan...', flush=True)
        pairs = conn.execute('SELECT COUNT(*) FROM (SELECT pre,post FROM edges NOT INDEXED GROUP BY pre,post)').fetchone()[0]
        annotations = [json.loads(a) for (a,) in conn.execute('SELECT annotations FROM neurons')]
        fields = sorted(set().union(*(a.keys() for a in annotations)))
        missing = {f: sum(f not in a for a in annotations) for f in fields}
        groups = {}
        for f in ('super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'top_nt', 'known_nt', 'side'):
            counts = {}
            for a in annotations:
                value = a.get(f, '(missing)')
                counts[value] = counts.get(value, 0) + 1
            groups[f] = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:20])
        # Candidate identification is explicitly annotation-based, not functional validation.
        rules = {
            'sensory_candidates': ('super_class', {'sensory', 'visual_projection', 'optic'}),
            'motor_output_candidates': ('super_class', {'motor', 'descending'}),
            'predicted_dopaminergic_candidates': ('top_nt', {'dopamine'}),
            'known_nt_exact_dopamine_candidates': ('known_nt', {'dopamine'}),
            'mushroom_body_candidates': ('cell_class', {'kenyon_cell', 'mbon', 'dan'}),
        }
        candidates = {name: {'field': field, 'exact_values_case_insensitive': sorted(values),
                             'count': sum(a.get(field, '').lower() in values for a in annotations),
                             'field_present_in_source': field in fields,
                             'status': 'annotation match only; requires circuit literature review'}
                      for name, (field, values) in rules.items()}
        isolated = conn.execute('''SELECT COUNT(*) FROM neurons n WHERE
            NOT EXISTS(SELECT 1 FROM edges WHERE pre=n.root_id) AND
            NOT EXISTS(SELECT 1 FROM edges WHERE post=n.root_id)''').fetchone()[0]
        print('Summarizing outgoing connectivity with a sequential graph scan...', flush=True)
        top = [{'root_id': str(i), 'outgoing_synapses': count} for i, count in conn.execute(
            'SELECT pre,SUM(syn_count) AS s FROM edges NOT INDEXED GROUP BY pre ORDER BY s DESC,pre LIMIT 10')]
        metadata = dict(conn.execute('SELECT key,value FROM metadata'))
    result = {'created_utc': stamp(), 'neurons': n, 'connection_region_rows': edges,
              'directed_neuron_pairs': pairs, 'anatomical_synapse_count': synapses,
              'isolated_neurons': isolated, 'neurons_without_any_annotations': sum(not a for a in annotations),
              'missing_annotations_by_field': missing, 'annotation_groups_top20': groups,
              'candidate_populations': candidates, 'top_outgoing': top,
              'source_manifest': json.loads(metadata['source_manifest']),
              'annotation_import': json.loads(metadata['annotation_import']),
              'database_bytes': Path(database).stat().st_size,
              'limitations': ['Connectivity is anatomical, not a validated dynamic brain model.',
                'Multiple neuropil rows can describe one neuron pair; rows are not individual synapses.',
                'Transmitter argmax is a prediction, not a validated synaptic sign or reward function.',
                'Annotation absence does not establish biological irrelevance. No circuits have been deleted.']}
    write_json(output, result)
    lines = ['# FlyBrain connectome inspection', '',
             f'- Proofread neurons: {n:,}', f'- Directed neuron pairs: {pairs:,}',
             f'- Connection-region rows: {edges:,}', f'- Anatomical synapses represented: {synapses:,}',
             f'- Isolated neurons: {isolated:,}', '', '## Candidate populations', '']
    lines += [f'- {key}: {value["count"]:,} ({value["status"]})' for key, value in candidates.items()]
    lines += ['', '## Missing annotations', '', '| Field | Missing neurons |', '|---|---:|']
    lines += [f'| {f} | {count:,} |' for f, count in missing.items()]
    lines += ['', '## Interpretation limits', ''] + ['- ' + v for v in result['limitations']]
    lines += ['', '## Sources', '', f'- Connectivity: {ZENODO}', f'- Annotations: {ANNOTATIONS}',
              f'- Exact annotation commit: `{result["source_manifest"]["annotation_commit"]}`',
              '', 'See the adjacent JSON report for source hashes, distributions, and import reconciliation.']
    Path(output).with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('neurons', 'directed_neuron_pairs', 'connection_region_rows', 'anatomical_synapse_count')}, indent=2))


def select_graph(args):
    if Path(args.output).exists():
        raise ValueError('Selection already exists; use a new output name to preserve its history.')
    with connect(args.database) as conn:
        if not conn.execute('SELECT 1 FROM neurons WHERE root_id=?', (args.seed,)).fetchone():
            raise ValueError('Seed is not a proofread neuron in this database')
        chosen = {args.seed}
        frontier = [args.seed]
        # Bounded breadth-first traversal ordered by connection strength, then ID.
        for neuron in frontier:
            if len(chosen) >= args.limit:
                break
            rows = conn.execute('''SELECT neighbor,SUM(syn_count) AS weight FROM (
                SELECT post AS neighbor,syn_count FROM edges WHERE pre=?
                UNION ALL SELECT pre AS neighbor,syn_count FROM edges WHERE post=?
                ) GROUP BY neighbor HAVING SUM(syn_count)>=? ORDER BY weight DESC,neighbor LIMIT ?''',
                (neuron, neuron, args.min_synapses, args.limit))
            for neighbor, _ in rows:
                if neighbor not in chosen:
                    chosen.add(neighbor)
                    frontier.append(neighbor)
                if len(chosen) >= args.limit:
                    break
        manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='source_manifest'").fetchone()[0])
    selection = {'schema_version': 2, 'created_utc': stamp(), 'source_manifest_sha256':
                 source_fingerprint(manifest),
                 'source_identity_policy': 'dataset, connectivity version, annotation commit, and source file SHA-256 hashes; excludes acquisition time',
                 'method': 'deterministic bidirectional breadth-first traversal; strongest neighbors first',
                 'seed': str(args.seed), 'requested_limit': args.limit,
                 'traversal_min_synapses': args.min_synapses,
                 'edge_policy': 'induced subgraph; retain ALL source edges between selected IDs',
                 'root_ids': [str(x) for x in sorted(chosen)],
                 'biological_status': 'engineering benchmark subset; not a validated functional circuit'}
    write_json(args.output, selection)
    print(f'Saved {len(chosen):,} IDs to {args.output}; source data unchanged.')


def benchmark(args):
    import numpy as np
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    ids = sorted({int(x) for x in selection['root_ids']})
    if not ids:
        raise ValueError('Cannot benchmark an empty selection')
    with connect(args.database) as conn:
        manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='source_manifest'").fetchone()[0])
        if selection.get('schema_version') == 1:
            expected = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        elif selection.get('schema_version') == 2:
            expected = source_fingerprint(manifest)
        else:
            raise ValueError('Unsupported selection schema version')
        if selection['source_manifest_sha256'] != expected:
            raise ValueError('Selection was made against different sources')
        # Temp table works on a read-only main database and avoids SQLite variable limits.
        conn.execute('CREATE TEMP TABLE selected(id INTEGER PRIMARY KEY)')
        conn.executemany('INSERT INTO selected VALUES (?)', ((i,) for i in ids))
        found = conn.execute('SELECT COUNT(*) FROM neurons JOIN selected ON root_id=id').fetchone()[0]
        if found != len(ids):
            raise ValueError('Selection contains IDs absent from this database')
        rows = conn.execute('''SELECT pre,post,SUM(syn_count) FROM edges e
            JOIN selected a ON a.id=e.pre JOIN selected b ON b.id=e.post
            GROUP BY pre,post ORDER BY pre,post''').fetchall()
    if not rows:
        raise ValueError('Selection has no internal connections')
    indices = {root: i for i, root in enumerate(ids)}
    pre = np.array([indices[r[0]] for r in rows], dtype=np.int32)
    post = np.array([indices[r[1]] for r in rows], dtype=np.int32)
    weights = np.array([r[2] for r in rows], dtype=np.float64)
    anatomical_synapses = int(weights.sum())
    incoming = np.bincount(post, weights=weights, minlength=len(ids))
    weights /= np.maximum(incoming[post], 1)
    rng = np.random.default_rng(args.seed)
    drive = np.zeros(len(ids), dtype=np.float64)
    drive[rng.choice(len(ids), size=max(1, len(ids) // 20), replace=False)] = 0.2

    def run():
        state = np.zeros(len(ids), dtype=np.float64)
        for _ in range(args.steps):
            current = np.bincount(post, weights=weights * state[pre], minlength=len(ids))
            state = 0.9 * state + 0.1 * np.tanh(current + drive)
        return state

    run()  # warm-up outside measured repetitions
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        state = run()
        times.append(time.perf_counter() - start)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite neural activity')
    median = float(np.median(times))
    result = {'created_utc': stamp(), 'model': 'unsigned normalized leaky tanh rate network',
              'model_parameters': {'retention': 0.9, 'update_fraction': 0.1, 'drive': 0.2,
                                   'driven_fraction': 0.05, 'normalization': 'incoming synapse count'},
              'limitations': ['All connections use positive weights for a propagation workload only.',
                 'No validated transmitter signs, time constants, plasticity, learning, or biological time scale.',
                 'CPU NumPy benchmark only; GPU performance is unmeasured.',
                 'Selection is an engineering subset, not a functioning organism.'],
              'neurons': len(ids), 'directed_pairs': len(rows),
              'anatomical_synapses_represented': anatomical_synapses,
              'steps': args.steps, 'repeats': args.repeats, 'rng_seed': args.seed,
              'elapsed_seconds': times, 'median_seconds': median,
              'steps_per_second': args.steps / median,
              'edge_updates_per_second': len(rows) * args.steps / median,
              'core_array_bytes': sum(a.nbytes for a in (pre, post, weights, incoming, state, drive)),
              'memory_note': 'Core array storage only; excludes Python/SQLite, temporary arrays and process peak RAM.',
              'final_activity_mean': float(state.mean()), 'final_activity_max': float(state.max()),
              'final_state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
              'selection_sha256': digest(args.selection), 'source_manifest_sha256': expected,
              'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
              'processor': platform.processor(), 'logical_cpu_count': os.cpu_count()}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('acquire', help='Download pinned public sources (~853 MB plus annotations)')
    p.add_argument('--annotation-commit', help='Full annotation Git SHA for reproducing a previous acquisition')
    p.set_defaults(func=acquire)
    p = sub.add_parser('verify', help='Verify original sources against SHA-256 manifest')
    p.set_defaults(func=lambda a: (verify(RAW), print('All original source hashes match.')))
    p = sub.add_parser('build', help='Build a new searchable SQLite database from preserved originals')
    p.add_argument('--output', type=Path, default=DB)
    p.set_defaults(func=lambda a: build_database(RAW, a.output))
    p = sub.add_parser('inspect', help='Write JSON and Markdown connectivity/annotation reports')
    p.add_argument('--output', type=Path, default=ROOT / 'reports' / 'inspection.json')
    p.set_defaults(func=lambda a: inspect_database(a.database, a.output))
    p = sub.add_parser('search', help='Search all annotation values using a literal substring')
    p.add_argument('text')
    p.add_argument('--limit', type=positive, default=10)
    p.set_defaults(func=search)
    p = sub.add_parser('select', help='Save a reversible induced subgraph as explicit neuron IDs')
    p.add_argument('--seed', type=positive, required=True)
    p.add_argument('--limit', type=positive, default=1000)
    p.add_argument('--min-synapses', type=positive, default=5)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=select_graph)
    p = sub.add_parser('benchmark', help='Benchmark CPU rate dynamics on a saved real subgraph')
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--steps', type=positive, default=500)
    p.add_argument('--repeats', type=positive, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=benchmark)
    for name, p in sub.choices.items():
        if name not in ('acquire', 'verify', 'build'):
            p.add_argument('--database', type=Path, default=DB)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
