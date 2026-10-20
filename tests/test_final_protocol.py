"""Prevent held-out leakage, silent source changes and repeat inference."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_final_evaluation import assert_hashes, claim_evaluation, cluster_interval, select_test_rows
from scripts.run_arcface_baseline import read_csv, sha256, write_csv


class FinalProtocolTests(unittest.TestCase):
    def split(self):
        return dict(halfface=dict(development=dict(known=['d'],unknown=[]),test_reserved=dict(known=['k'],unknown=['u'])),
                    icfd_sample=dict(engineering_development=['i0'],reserved=['i1']))

    def rows(self):
        hf = [dict(dataset='halfface',subject_id=s,path=p,sha256=p,mask_status=m,pose=pose)
              for s,p,m,pose in [('d','dev','unmasked','front'),('k','front','unmasked','front'),
                                ('k','side','unmasked','left'),('k','mask','masked','front'),('u','unk','unmasked','front')]]
        icfd = [dict(dataset='icfd_sample',subject_id=s,path=p,sha256=p,source_role=role)
                for s,p,role in [('i0','idev','gallery'),('i1','igal','gallery'),('i1','iprobe','probe')]]
        return hf,icfd

    def test_only_reserved_ids_and_separate_gallery_probes(self):
        rows = select_test_rows(*self.rows(),self.split())
        self.assertEqual({r['subject_id'] for r in rows},{'k','u','i1'})
        self.assertEqual({r['path'] for r in rows if r['evaluation_role'] == 'probe'},{'mask','unk','iprobe'})
        self.assertEqual({r['path'] for r in rows if r['evaluation_role'] == 'gallery'},{'front','side','igal'})

    def test_partition_overlap_is_rejected(self):
        split = self.split()
        split['halfface']['development']['known'] = ['k']
        with self.assertRaisesRegex(ValueError,'partition overlap'):
            select_test_rows(*self.rows(),split)

    def test_mixed_dataset_metadata_can_be_saved_together(self):
        rows = select_test_rows(*self.rows(),self.split())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.csv'
            write_csv(path,rows)
            restored = read_csv(path)
            self.assertEqual(len(restored),len(rows))
            self.assertEqual({r['dataset'] for r in restored},{'halfface','icfd_sample'})

    def test_duplicate_image_across_development_and_test_is_rejected(self):
        hf,icfd = self.rows()
        hf[0]['sha256'] = 'mask'
        with self.assertRaisesRegex(ValueError,'overlap with development'):
            select_test_rows(hf,icfd,self.split())

    def test_changed_frozen_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'settings.json'
            path.write_text('original')
            hashes = {'settings.json':sha256(path)}
            assert_hashes(hashes,root)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):
                assert_hashes(hashes,root)

    def test_second_inference_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'freeze.json').write_text(json.dumps({'threshold':.5}))
            claim_evaluation(path)
            with self.assertRaises(FileExistsError):
                claim_evaluation(path)

    def test_identical_identity_outcomes_give_fixed_cluster_interval(self):
        rows = [dict(subject_id=s,truth_status='known',correct=correct) for s in ['a','b'] for correct in [True,False]]
        self.assertEqual(cluster_interval(rows,'known',lambda r:r['correct'],lambda r:1,100,42),[.5,.5])


if __name__ == '__main__':
    unittest.main()
