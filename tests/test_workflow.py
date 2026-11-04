# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 FlyBrain contributors
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from flybrain.cli import benchmark, build_database, digest, inspect_database, select_graph, source_fingerprint, verify, write_json


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        # Large integer IDs must remain exact, including in exported JSON.
        self.ids = [720575940000000001 + i for i in range(4)]
        np.save(self.raw / 'proofread_root_ids_783.npy', np.array(self.ids, dtype=np.int64))
        a, b, c, d = self.ids
        columns = {'pre_pt_root_id': [a, a, b], 'post_pt_root_id': [b, b, c],
                   'neuropil': ['A', 'B', 'A'], 'syn_count': [3, 4, 3]}
        for nt in ('gaba', 'ach', 'glut', 'oct', 'ser', 'da'):
            columns[nt + '_avg'] = [0.9 if nt == 'ach' else 0.02] * 3
        feather.write_feather(pa.table(columns), self.raw / 'proofread_connections_783.feather')
        (self.raw / 'Supplemental_file1_neuron_annotations.tsv').write_text(
            f'root_id\tsuper_class\tcell_class\ttop_nt\n{a}\tsensory\t\tacetylcholine\n'
            f'{b}\tcentral\tKenyon_Cell\tdopamine\n{c}\tdescending\t\tgaba\n', encoding='utf-8')
        write_json(self.raw / 'manifest.json', {'dataset': 'SYNTHETIC TEST FIXTURE',
            'annotation_commit': 'fixture-only', 'files': {
                f.name: {'sha256': digest(f)} for f in self.raw.iterdir()}})
        self.db = self.root / 'test.sqlite'
        with redirect_stdout(io.StringIO()):
            build_database(self.raw, self.db)

    def selection(self, limit=2):
        path = self.root / f'selection-{limit}.json'
        with redirect_stdout(io.StringIO()):
            select_graph(argparse.Namespace(database=self.db, output=path,
                seed=self.ids[0], limit=limit, min_synapses=5))
        return path

    def test_report_distinguishes_rows_pairs_synapses_and_isolates(self):
        output = self.root / 'report.json'
        with redirect_stdout(io.StringIO()):
            inspect_database(self.db, output)
        report = json.loads(output.read_text())
        self.assertEqual((report['neurons'], report['connection_region_rows'],
            report['directed_neuron_pairs'], report['anatomical_synapse_count']), (4, 3, 2, 10))
        self.assertEqual(report['isolated_neurons'], 1)
        self.assertEqual(report['neurons_without_any_annotations'], 1)
        self.assertEqual(report['candidate_populations']['mushroom_body_candidates']['count'], 1)
        self.assertEqual(report['candidate_populations']['predicted_dopaminergic_candidates']['count'], 1)
        self.assertFalse(report['candidate_populations']['known_nt_exact_dopamine_candidates']['field_present_in_source'])

    def test_selection_is_reversible_and_preserves_integer_ids(self):
        before = digest(self.db)
        path = self.selection()
        self.assertEqual(json.loads(path.read_text())['root_ids'], [str(x) for x in self.ids[:2]])
        self.assertEqual(before, digest(self.db))
        with self.assertRaises(ValueError):
            self.selection()

    def test_source_tampering_is_detected(self):
        with (self.raw / 'proofread_root_ids_783.npy').open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            verify(self.raw)

    def test_benchmark_aggregates_neuropils_and_matches_one_step_equation(self):
        path = self.selection()
        output = self.root / 'benchmark.json'
        args = argparse.Namespace(database=self.db, selection=path, output=output,
                                  steps=1, repeats=2, seed=42)
        with redirect_stdout(io.StringIO()):
            benchmark(args)
        result = json.loads(output.read_text())
        self.assertEqual(result['directed_pairs'], 1)
        self.assertEqual(result['anatomical_synapses_represented'], 7)
        self.assertAlmostEqual(result['final_activity_max'], 0.1 * np.tanh(0.2))
        with redirect_stdout(io.StringIO()):
            benchmark(args)
        self.assertEqual(result['final_state_sha256'], json.loads(output.read_text())['final_state_sha256'])

    def test_benchmark_rejects_source_mismatch(self):
        path = self.selection()
        selection = json.loads(path.read_text())
        selection['source_manifest_sha256'] = 'wrong'
        write_json(path, selection)
        with self.assertRaisesRegex(ValueError, 'different sources'):
            benchmark(argparse.Namespace(database=self.db, selection=path))

    def test_build_never_overwrites_existing_database(self):
        before = digest(self.db)
        with self.assertRaises(ValueError):
            build_database(self.raw, self.db)
        self.assertEqual(before, digest(self.db))

    def test_source_identity_is_portable_across_download_times(self):
        original = json.loads((self.raw / 'manifest.json').read_text())
        other = dict(original, created_utc='a different acquisition time')
        self.assertEqual(source_fingerprint(original), source_fingerprint(other))
        other = dict(other, annotation_commit='a-different-revision')
        self.assertNotEqual(source_fingerprint(original), source_fingerprint(other))


if __name__ == '__main__':
    unittest.main()
