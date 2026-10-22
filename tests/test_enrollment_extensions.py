import numpy as np
import pytest

from src.evaluation.enrollment_extensions import (enrollment_variant,gallery_scenarios,
    check_development_ids,probe_rows,score_rows)


def test_variants_deterministic_distinct_and_source_preserved():
    source=np.random.default_rng(9).integers(0,256,(112,112,3),dtype=np.uint8)
    original=source.copy()
    for name in ['blur_sigma1','resize56','jpeg50','gamma0.7','gamma1.4']:
        a=enrollment_variant(source,name); b=enrollment_variant(source,name)
        assert a.shape==source.shape and a.dtype==source.dtype
        assert not np.array_equal(a,source)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(source,original)
    with pytest.raises(ValueError): enrollment_variant(source[:96], 'resize56')
    with pytest.raises(ValueError): enrollment_variant(source,'random_unregistered_transform')


def test_gamma_semantics_are_explicit():
    source=np.full((112,112,3),100,dtype=np.uint8)
    assert enrollment_variant(source,'gamma0.7').mean()>100
    assert enrollment_variant(source,'gamma1.4').mean()<100


def test_growth_retains_core_and_only_varies_background_composition():
    ids=[f'id{i:02d}' for i in range(55)]
    core,scenarios=gallery_scenarios(ids,[42,123,2026,7,99])
    assert core==ids[:10]
    assert len(scenarios)==7
    for scenario in scenarios:
        assert len(scenario['ids'])==scenario['size']
        assert set(core)<=set(scenario['ids'])<=set(ids)
    assert len({tuple(s['ids']) for s in scenarios if s['size']==25})==5
    with pytest.raises(ValueError): gallery_scenarios(ids[:54],[42])


def test_development_guard_rejects_reserved_and_foreign_inputs():
    good=dict(dataset='halfface',subject_id='a')
    check_development_ids([good],['a'],['u'],['test'])
    with pytest.raises(ValueError): check_development_ids([good],['a'],['u'],['a'])
    with pytest.raises(ValueError): check_development_ids([dict(good,subject_id='test')],['a'],['u'],['test'])


def test_fixed_probes_exclude_background_gallery_people():
    rows=[dict(subject_id='core',mask_status='masked'),dict(subject_id='core',mask_status='unmasked'),
          dict(subject_id='background',mask_status='masked'),dict(subject_id='unknown',mask_status='unmasked')]
    selected=probe_rows(rows,['core'],['unknown'])
    assert [r['subject_id'] for r in selected]==['core','unknown']


def test_failed_probes_remain_and_unknowns_are_never_rank1_correct():
    common=dict(dataset='halfface',subject_id='u',truth_status='unknown',quality='.9')
    rows=[dict(common,path='a',status='ok',embedding_index='0'),dict(common,path='b',status='no_face',embedding_index='-1')]
    out=score_rows(rows,np.array([[1.,0.]]),['x','y'],np.eye(2))
    assert len(out)==2 and not any(r['rank1_correct'] for r in out)
    assert out[0]['top1_id']=='x' and out[1]['top1_score']==''
