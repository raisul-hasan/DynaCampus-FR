"""Local headless Chrome checks through its browser protocol, no native UI input."""
import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/demo/dynamic_development_20261019T233918475526Z'
ASSETS=ROOT/'paper_assets/development/demo_v1'


class Browser:
    def __init__(self):
        tabs=json.load(urllib.request.urlopen('http://127.0.0.1:9227/json'))
        page=next(t for t in tabs if t.get('type')=='page' and t.get('url','').startswith('http://127.0.0.1:8517'))
        self.ws=connect(page['webSocketDebuggerUrl'],max_size=None,proxy=None,legacy=True)
        self.index=0
        self.call('Page.enable')
        self.call('Runtime.enable')
        self.call('Emulation.setDeviceMetricsOverride',width=1440,height=1100,deviceScaleFactor=1,mobile=False)

    def call(self,method,**params):
        self.index+=1
        index=self.index
        self.ws.send(json.dumps(dict(id=index,method=method,params=params)))
        while True:
            result=json.loads(self.ws.recv(timeout=30))
            if result.get('id')==index:
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result.get('result',{})

    def evaluate(self,expression):
        result=self.call('Runtime.evaluate',expression=expression,returnByValue=True,awaitPromise=True)
        if 'exceptionDetails' in result:
            raise RuntimeError(result['exceptionDetails'])
        return result['result'].get('value')

    def observe(self):
        return self.evaluate("""JSON.stringify({text:document.body.innerText,
          sidebar:document.querySelector('[data-testid=stSidebar]')?.getBoundingClientRect().toJSON(),
          inputs:Array.from(document.querySelectorAll('input')).map(e=>({type:e.type,id:e.id,placeholder:e.placeholder,
            label:Array.from(e.labels||[]).map(l=>l.innerText).join(' '),checked:e.checked,value:e.type==='file'?undefined:e.value})),
          buttons:Array.from(document.querySelectorAll('button')).map(e=>e.innerText).filter(Boolean)})""")

    def wait_for(self,text,timeout=30):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if self.evaluate('document.body.innerText').find(text)>=0:
                return
            time.sleep(.25)
        raise AssertionError(f'Browser did not show {text!r}')

    def screenshot(self,name):
        ASSETS.mkdir(parents=True,exist_ok=True)
        time.sleep(1)
        self.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(true))))')
        data=self.call('Page.captureScreenshot',format='png',captureBeyondViewport=True,fromSurface=True)
        (ASSETS/name).write_bytes(base64.b64decode(data['data']))

    def click_label(self,text):
        expression="""(()=>{const e=Array.from(document.querySelectorAll('input')).find(e=>Array.from(e.labels||[]).some(l=>l.innerText.includes(TEXT)));if(!e)throw Error('No labelled input');e.click();return true})()"""
        self.evaluate(expression.replace('TEXT',json.dumps(text)))

    def click_button(self,text):
        expression="""(()=>{const e=Array.from(document.querySelectorAll('button')).find(e=>e.innerText.includes(TEXT));if(!e)throw Error('No button');e.click();return true})()"""
        self.evaluate(expression.replace('TEXT',json.dumps(text)))

    def type_field(self,label,text):
        expression="""(()=>{const e=Array.from(document.querySelectorAll('input[type=text]')).find(e=>Array.from(e.labels||[]).some(l=>l.innerText.includes(LABEL)));if(!e)throw Error('No text field');e.focus();return true})()"""
        self.evaluate(expression.replace('LABEL',json.dumps(label)))
        self.call('Input.insertText',text=text)
        self.call('Input.dispatchKeyEvent',type='keyDown',key='Enter',code='Enter',windowsVirtualKeyCode=13)
        self.call('Input.dispatchKeyEvent',type='keyUp',key='Enter',code='Enter',windowsVirtualKeyCode=13)

    def upload(self,key):
        fixture=json.loads((RUN/'demo_fixture.json').read_text())
        paths=fixture['enrollment_paths'] if key=='enrollment' else [fixture[key]]
        doc=self.call('DOM.getDocument')
        node=self.call('DOM.querySelector',nodeId=doc['root']['nodeId'],selector='input[type=file]')['nodeId']
        if not node:
            raise RuntimeError('No file upload control on current page')
        self.call('DOM.setFileInputFiles',nodeId=node,files=[str(ROOT/p) for p in paths])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['observe','label','button','text','upload','screenshot','reload'])
    parser.add_argument('value',nargs='?')
    parser.add_argument('extra',nargs='?')
    args=parser.parse_args()
    browser=Browser()
    if args.action=='label': browser.click_label(args.value)
    elif args.action=='button': browser.click_button(args.value)
    elif args.action=='text': browser.type_field(args.value,args.extra)
    elif args.action=='upload': browser.upload(args.value)
    elif args.action=='screenshot': browser.screenshot(args.value)
    elif args.action=='reload': browser.call('Page.reload')
    if args.action!='observe': time.sleep(1)
    observation=browser.observe()
    with (RUN/'browser_events.jsonl').open('a',encoding='utf-8') as output:
        output.write(json.dumps(dict(action=args.action,value=args.value,observation=json.loads(observation)))+'\n')
    state=json.loads(observation)
    print(observation if args.action=='observe' else json.dumps(dict(text=state['text'],buttons=state['buttons'])))
