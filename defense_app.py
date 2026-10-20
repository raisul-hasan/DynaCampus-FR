"""Defense case viewer and isolated real-inference gallery walkthrough."""
import json
from pathlib import Path
import streamlit as st
from src.defense_workflow import ROOT, STEPS, DefenseWorkflow, default_run

st.set_page_config(page_title='DynaCampus-FR | Defense evidence', layout='wide')
run = default_run()
manifest = json.loads((run/'case_manifest.json').read_text(encoding='utf-8'))
st.title('DynaCampus-FR: defense evidence')
st.caption('Development examples and real-image demonstration. The frozen final evaluation is unchanged.')
section = st.sidebar.radio('Evidence', ['Paired cases', 'Live gallery walkthrough', 'Study results'])
show_photos = st.sidebar.checkbox('Show local source photographs', value=False)
st.sidebar.caption('Photographs remain local. Saved public evidence cards contain no portraits.')

if section == 'Paired cases':
    labels = {f"{c['case_id']} | {c['model']} | {c['category'].replace('_',' ')}": c for c in manifest['cases']}
    choice = st.selectbox('Case', list(labels))
    case = labels[choice]
    image = ROOT/manifest['private_directory']/'cards'/f"{case['case_id']}.png" if show_photos else ROOT/'paper_assets/development/defense_v1'/f"fig_case_{case['case_id']}.png"
    st.image(str(image), width='stretch')
    st.info('Fixed-rule cases keep the score and margin thresholds identical. Separately calibrated cases change thresholds as well as enrollment policy.')
    with st.expander('Provenance and full-precision values'):
        st.json(case)

elif section == 'Live gallery walkthrough':
    st.subheader('Real inference in an isolated demonstration gallery')
    st.caption('54 background identities use verified cached development templates. Target enrollment photos and every probe are processed afresh through SCRFD and ArcFace.')
    if 'defense_workflow' not in st.session_state:
        if st.button('Start isolated walkthrough', type='primary'):
            with st.spinner('Load real models and initialize a separate gallery...'):
                st.session_state.defense_workflow = DefenseWorkflow(run)
            st.rerun()
    else:
        workflow = st.session_state.defense_workflow
        a, b, c = st.columns(3)
        a.metric('Gallery identities', workflow.gallery.size)
        b.metric('Completed steps', workflow.step)
        c.metric('Target enrolled', 'Yes' if workflow.fixture['subject_id'] in workflow.gallery._embeddings else 'No')
        st.caption('Fixed score 0.377859 | margin 0.15 | quality gate disabled | no retraining')
        if workflow.step < len(STEPS):
            st.write(f'Next: {STEPS[workflow.step][1]}')
            if st.button('Run next walkthrough step', type='primary'):
                with st.spinner('Execute and verify...'):
                    st.session_state.last_defense_event = workflow.next_step()
                st.rerun()
        else:
            st.success('All ten steps complete. The target has been removed; the 54 background identities remain in this isolated gallery.')
        if 'last_defense_event' in st.session_state:
            event = st.session_state.last_defense_event
            if event.get('decision'):
                st.subheader(event['decision'])
                st.write('Accepted identity: '+(event.get('accepted_id') or 'None'))
                if event.get('face_count'):
                    first, second, gap = st.columns(3)
                    first.metric('Best similarity', f"{event['top1_score']:.6f}")
                    second.metric('Second similarity', f"{event['top2_score']:.6f}")
                    gap.metric('Top-two margin', f"{event['margin']:.6f}")
                    st.caption(f"Candidates: {event['top1_id']} and {event['top2_id']}. A close candidate is not an accepted identity.")
                else:
                    st.warning('No usable face detection. Recognition has no embedding or match score for this image.')
            elif event['action'] == 'enroll':
                st.success(f"Enrolled the target from {event['photos']} original photos. The recognition checkpoints are unchanged.")
            elif event['action'] == 'reload':
                st.success('Gallery reloaded from disk with exact template equality.')
            elif event['action'] == 'remove':
                st.success('Only the demonstration target was removed from this isolated gallery.')
            with st.expander('Verification details and full-precision event'):
                st.json(event)
            if show_photos and event.get('source_path'):
                st.image(str(ROOT/event['source_path']), width=240)
        st.dataframe([dict(step=r['sequence'], action=r['action'], decision=r.get('decision',''),
                           accepted_id=r.get('accepted_id',''), gallery=r['gallery_size']) for r in workflow.events], hide_index=True)
        st.caption('CAPTURE_FAILURE is an extraction outcome, separate from the four decision states. Event logs and the isolated gallery are saved locally.')

else:
    st.subheader('Frozen final evaluation')
    st.image(str(ROOT/'paper_assets/heldout/final_v1/fig01_heldout_comparison.png'), width='stretch')
    st.caption('Primary: 151/160 correct known acceptances; 0/151 observed false known accepts. Zero observed errors is not a population guarantee.')
    st.subheader('Additional development comparison')
    st.image(str(ROOT/'paper_assets/development/backbones_v1/fig01_backbone_comparison.png'), width='stretch')
    st.caption('Calibration and evaluation reuse development images. Checkpoints differ in pretraining and runtime engines. The enrollment benefit depends on the decision rule.')
