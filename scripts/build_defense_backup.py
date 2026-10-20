"""Build a clearly labelled evidence video from case cards and prior real captures.

This is an assembled evidence walkthrough, never a new live screen recording.
Uses a downloaded official-PyPI imageio-ffmpeg wheel without installing packages.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import sha256
from scripts.prepare_defense_cases import font
from src.defense_workflow import default_run


def main():
    run=default_run()
    manifest=json.loads((run/'case_manifest.json').read_text())
    private=ROOT/manifest['private_directory']
    frames=private/'backup_frames'
    frames.mkdir(exist_ok=True)
    wheel=next((ROOT/'data/tools').glob('imageio_ffmpeg-*-win_amd64.whl'),None)
    if wheel is None:
        raise FileNotFoundError('Download imageio-ffmpeg with pip download --no-deps --dest data/tools')
    binary=ROOT/'data/tools/defense_ffmpeg.exe'
    with zipfile.ZipFile(wheel) as archive:
        name=next(n for n in archive.namelist() if n.endswith('.exe') and '/binaries/' in n)
        binary.write_bytes(archive.read(name))
    slides=[]
    def add(title,subtitle,source=None,duration=7,body=None):
        image=Image.new('RGB',(1920,1080),'#edf2f7')
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,1920,125),fill='#153247')
        draw.text((45,20),title,fill='white',font=font(31))
        draw.text((45,72),subtitle,fill='#d4e4f0',font=font(23))
        if source:
            picture=Image.open(source).convert('RGB')
            picture.thumbnail((1830,855))
            image.paste(picture,((1920-picture.width)//2,140+(855-picture.height)//2))
        if body:
            for i,line in enumerate(body):
                draw.text((80,200+i*72),line,fill='#153247',font=font(32))
        draw.text((45,1030),'Assembled evidence walkthrough: saved cards and prior genuine browser screenshots. No live capture is implied.',fill='#4a6173',font=font(22))
        path=frames/f'frame_{len(slides):03d}.png'
        image.save(path)
        slides.append(dict(frame=path.name,duration_seconds=duration,title=title,source=str(source.relative_to(ROOT)) if source else None,
                           source_sha256=sha256(source) if source else None))
    add('DynaCampus-FR: defense backup evidence','Development examples, frozen final results and executable gallery workflow',duration=8,
        body=['This video is assembled from saved evidence; it is not a fresh screen recording.',
              'Examples include improvements, harms, unknown rejection and a false accept.',
              'Full source paths, hashes and candidate selection are recorded locally.',
              'Live walkthrough: python -m streamlit run defense_app.py',
              'All original frozen final settings and predictions remain unchanged.'])
    add('The frozen final comparison','Same 160 known attempts and 151 unknown attempts',ROOT/'paper_assets/heldout/final_v1/fig01_heldout_comparison.png',9)
    add('Additional development evidence','Enrollment effects depend on the checkpoint and calibrated decision rule',ROOT/'paper_assets/development/backbones_v1/fig01_backbone_comparison.png',9)
    for case in manifest['cases']:
        add(case['case_id']+': '+case['category'].replace('_',' '),case['model']+' | '+case['scope']+' | source photos omitted from this backup',
            ROOT/'paper_assets/development/defense_v1'/f"fig_case_{case['case_id']}.png",7)
    capture_source=ROOT/'results/demo/dynamic_development_20261019T233918475526Z'
    for stem,title in [('fig03_app_enrollment','Actual prior browser capture: enrollment'),('fig04_app_known','Actual prior browser capture: known masked probe'),
                       ('fig05_app_unknown','Actual prior browser capture: unknown probe'),('fig06_app_persistence','Actual prior browser capture: reload'),
                       ('fig07_app_after_removal','Actual prior browser capture: removal')]:
        source=ROOT/'paper_assets/development/demo_v1'/(stem+'.png')
        add(title,'Recorded 20 October 2026; one-identity example. This is not the new 55-identity ambiguity walkthrough.',source,6)
    add('What the evidence supports','Measured prototype behavior, with explicit limits',duration=9,
        body=['Few-photo enrollment helped under the frozen final comparison.',
              'Additional development tests show that the benefit is conditional.',
              'Gallery insertion, reload and removal execute without backbone retraining.',
              'Zero observed false accepts is a limited-sample result.',
              'Real CCTV, temporal tracking and deployment reliability remain unvalidated.'])
    listing=''.join(f"file '{r['frame']}'\nduration {r['duration_seconds']}\n" for r in slides)+f"file '{slides[-1]['frame']}'\n"
    (frames/'concat.txt').write_text(listing,encoding='utf-8')
    output=private/'defense_evidence_backup.mp4'
    command=[str(binary),'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i','concat.txt',
             '-vf','fps=10,format=yuv420p','-c:v','libx264','-preset','fast','-crf','20','-movflags','+faststart',
             '-t',str(sum(r['duration_seconds'] for r in slides)),str(output)]
    subprocess.run(command,cwd=frames,check=True)
    # Decode representative frames at every chapter boundary as a real artifact check.
    import cv2
    video=cv2.VideoCapture(str(output))
    if not video.isOpened():
        raise RuntimeError('Encoded backup cannot be opened')
    fps=video.get(cv2.CAP_PROP_FPS)
    actual_duration=video.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    elapsed=0
    for slide in slides:
        video.set(cv2.CAP_PROP_POS_MSEC,(elapsed+1)*1000)
        ok,frame=video.read()
        if not ok or frame.shape[:2]!=(1080,1920):
            raise RuntimeError('Backup chapter failed decoding')
        elapsed+=slide['duration_seconds']
    video.release()
    if abs(actual_duration-elapsed)>.2:
        raise RuntimeError('Backup duration does not match the chapter manifest')
    receipt=dict(output=output.relative_to(ROOT).as_posix(),type='assembled_evidence_walkthrough_not_live_screen_recording',
        reused_prior_browser_captures=True,new_browser_capture=False,portraits_in_backup=False,chapters=slides,
        expected_duration_seconds=elapsed,decoded_duration_seconds=actual_duration,decoded_fps=fps,
        source_capture_run=capture_source.relative_to(ROOT).as_posix(),video_sha256=sha256(output),
        encoder_wheel_sha256=sha256(wheel),encoder_sha256=sha256(binary),script_sha256=sha256(Path(__file__)),
        all_chapters_decode=True)
    (run/'backup_video.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='chapters'},indent=2))


if __name__=='__main__':
    main()
