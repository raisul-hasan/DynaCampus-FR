import tempfile
import unittest
import numpy as np
from src.gallery import FaceGallery
from src.demo_workflow import recognize


class DynamicGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gallery = FaceGallery(self.temp.name)
        self.a,self.b = np.eye(512,dtype=np.float32)[:2]

    def tearDown(self):
        self.temp.cleanup()

    def test_insert_search_reload_remove(self):
        self.assertEqual(recognize(self.gallery,self.a,1.)['decision'],'UNKNOWN')
        self.assertTrue(self.gallery.enroll('S001','Example',self.a))
        self.assertEqual(recognize(self.gallery,self.a,1.)['accepted_id'],'S001')
        reloaded = FaceGallery(self.temp.name)
        self.assertEqual(reloaded.search(self.a)['top1_id'],'S001')
        self.assertTrue(reloaded.remove('S001'))
        self.assertEqual(FaceGallery(self.temp.name).size,0)

    def test_multi_photo_default_is_unweighted_mean(self):
        self.assertTrue(self.gallery.enroll_multi('S001','Example',[self.a,self.b],[.1,.9]))
        np.testing.assert_allclose(self.gallery._embeddings['S001'],(self.a+self.b)/np.sqrt(2),atol=1e-7)

    def test_invalid_vectors_cannot_corrupt_gallery(self):
        self.gallery.enroll('S001','Example',self.a)
        for vector in [np.zeros(512),np.ones(3),np.full(512,np.nan),np.full(512,np.inf)]:
            self.assertFalse(self.gallery.enroll('S002','Bad',vector))
        self.assertEqual(self.gallery.size,1)
        self.assertIsNone(self.gallery.search(np.zeros(512))['top1_id'])

    def test_pending_identity_is_unsearchable_until_approved(self):
        self.gallery.enroll('AutoID_1','Pending',self.a)
        self.assertEqual(self.gallery.size,0)
        self.assertIsNone(self.gallery.search(self.a)['top1_id'])
        self.assertEqual(len(self.gallery.list_pending()),1)
        self.assertTrue(self.gallery.approve_pending('AutoID_1','Approved'))
        self.assertEqual(self.gallery.list_pending(),[])
        self.assertEqual(FaceGallery(self.temp.name).size,1)

    def test_search_preserves_boundary_precision_and_reports_backend(self):
        vector = .12345678*self.a+np.sqrt(1-.12345678**2)*self.b
        self.gallery.enroll('S001','Example',self.a)
        result = self.gallery.search(vector,threshold=.12345)
        self.assertNotEqual(result['top1_score'],round(result['top1_score'],4))
        self.assertEqual(result['search_backend'],'NumPy exact cosine')

    def test_updating_approved_auto_identity_keeps_approval(self):
        self.gallery.enroll('AutoID_1','Pending',self.a)
        self.gallery.approve_pending('AutoID_1','Approved')
        self.gallery.enroll('AutoID_1','Updated',self.b)
        self.assertEqual(self.gallery.size,1)
        self.assertEqual(self.gallery.list_pending(),[])

    def test_close_candidates_are_ambiguous_without_accepted_identity(self):
        self.gallery.enroll('S001','A',self.a)
        self.gallery.enroll('S002','B',.99*self.a+.1*self.b)
        result = recognize(self.gallery,self.a,1.)
        self.assertEqual(result['decision'],'AMBIGUOUS')
        self.assertIsNone(result['accepted_id'])

    def test_pathlike_subject_ids_are_rejected(self):
        self.assertFalse(self.gallery.enroll('../outside','Bad',self.a))


if __name__ == '__main__':
    unittest.main()
