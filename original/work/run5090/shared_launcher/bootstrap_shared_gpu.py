"""Private pinned Ollama and content-addressed models; no changes to user tags."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import time
from urllib.request import Request, urlopen
import zipfile

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def download(url,path,expected,size=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if (size is None or path.stat().st_size==size) and sha(path)==expected:return 0
        path.rename(path.with_name(path.name+'.corrupt-'+str(time.time_ns())))
    part=path.with_name(path.name+'.part')
    offset=part.stat().st_size if part.exists() else 0
    if size is not None and offset>=size:
        if offset==size and sha(part)==expected:part.replace(path);return 0
        part.rename(part.with_name(part.name+'.corrupt-'+str(time.time_ns())));offset=0
    headers={'User-Agent':'GraphRAG-reproducibility/1'}
    if offset:headers['Range']=f'bytes={offset}-'
    print(f'Download {path.name}, resume={offset}',flush=True)
    with urlopen(Request(url,headers=headers),timeout=120) as response:
        if offset and response.status!=206:offset=0
        if offset and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
            raise RuntimeError('invalid HTTP range; file not appended')
        transferred=0;notified=0
        with part.open('ab' if offset else 'wb') as out:
            while block:=response.read(4*1024*1024):
                out.write(block);transferred+=len(block)
                if transferred-notified>=512*1024*1024:
                    print(f'  {path.name}: {offset+transferred} bytes',flush=True);notified=transferred
    if (size is not None and part.stat().st_size!=size) or sha(part)!=expected:
        if size is None or part.stat().st_size>=size:
            part.rename(part.with_name(part.name+'.corrupt-'+str(time.time_ns())))
        raise RuntimeError('download size/SHA mismatch: '+path.name)
    part.replace(path);return transferred


def safe_extract(archive,directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as src:
        for entry in src.infolist():
            relative=PurePosixPath(entry.filename.replace('\\','/'))
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):raise ValueError('unsafe archive entry')
            target=directory.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(directory):raise ValueError('archive entry escapes runtime')
            if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with src.open(entry) as source,target.open('wb') as out:shutil.copyfileobj(source,out,4*1024*1024)


def gpu_inventory():
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    rows=[]
    for line in result.stdout.splitlines():
        uuid,name,total,free,driver=[x.strip() for x in line.split(',')]
        rows.append(dict(uuid=uuid,name=name,total_mib=int(total),free_mib=int(free),driver=driver))
    return rows


def select_gpu(rows):
    candidates=[row for row in rows if row['total_mib']>=22000]
    if not candidates:raise RuntimeError('Main models need the 5090 host (at least 22 GiB VRAM); no CPU fallback attempted')
    return sorted(candidates,key=lambda row:('5090' not in row['name'], -row['total_mib']))[0]


def model_store(extra=None):
    identities=read(ROOT/'runtime/model_manifests/IDENTITY.json')
    sources=[]
    for candidate in [extra,os.environ.get('OLLAMA_MODELS'),Path.home()/'.ollama/models']:
        if candidate and Path(candidate).exists():sources.append(Path(candidate).expanduser().resolve())
    destination=ROOT/'runtime/ollama_models';blobs=destination/'blobs';blobs.mkdir(parents=True,exist_ok=True)
    downloaded=0;linked=0;copied=0;seen=set()
    for identity in identities:
        role=identity['role'];manifest_path=ROOT/'runtime/model_manifests'/(role+'.json')
        if sha(manifest_path)!=identity['model']['digest']:raise RuntimeError('model manifest altered')
        manifest=read(manifest_path);name,tag=identity['model']['name'].split(':',1)
        for layer in [manifest['config'],*manifest['layers']]:
            digest=layer['digest'];expected=digest.removeprefix('sha256:')
            if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):raise ValueError('invalid layer digest')
            if digest in seen:continue
            seen.add(digest);filename=digest.replace(':','-');target=blobs/filename
            if target.exists() and target.stat().st_size==layer['size'] and sha(target)==expected:continue
            existing=next((store/'blobs'/filename for store in sources if (store/'blobs'/filename).is_file()
                           and (store/'blobs'/filename).stat().st_size==layer['size'] and sha(store/'blobs'/filename)==expected),None)
            if existing is not None and not target.exists():
                try:os.link(existing,target);linked+=layer['size']
                except OSError:
                    if shutil.disk_usage(destination).free<layer['size']+10*1024**3:raise RuntimeError('not enough free space for model copy plus results')
                    shutil.copyfile(existing,target);copied+=layer['size']
            else:
                if shutil.disk_usage(destination).free<layer['size']+10*1024**3:raise RuntimeError('not enough free space for model plus results')
                downloaded+=download(f'https://registry.ollama.ai/v2/library/{name}/blobs/{digest}',target,expected,layer['size'])
        target=destination/'manifests/registry.ollama.ai/library'/name/tag;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=manifest_path.read_bytes():raise RuntimeError('private model tag changed')
        if not target.exists():target.write_bytes(manifest_path.read_bytes())
    return destination,dict(downloaded_bytes=downloaded,hardlinked_bytes=linked,copied_bytes=copied)


def process_fingerprint(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return None
    try:
        size=wintypes.DWORD(32768);buffer=ctypes.create_unicode_buffer(32768)
        kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
        if not kernel.QueryFullProcessImageNameW(handle,0,buffer,ctypes.byref(size)):return None
        times=[wintypes.FILETIME() for _ in range(4)]
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return None
        created=(times[0].dwHighDateTime<<32)+times[0].dwLowDateTime
        return dict(pid=pid,image=str(Path(buffer.value).resolve()),created=created)
    finally:kernel.CloseHandle(handle)


def stop_owned(fingerprint):
    if not fingerprint or process_fingerprint(fingerprint['pid'])!=fingerprint:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
    handle=kernel.OpenProcess(1|0x100000,False,fingerprint['pid'])
    if not handle:return False
    try:return bool(kernel.TerminateProcess(handle,0))
    finally:kernel.CloseHandle(handle)


@contextmanager
def server5090(work,extra_models=None):
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from ragsel.k1_study_execution import StrictOllama,atomic_json
    from ragsel.k1_inference import ModelRef
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    gpu=select_gpu(gpu_inventory())
    old=work/'server_identity.json'
    if old.exists():
        previous=read(old)
        if previous.get('fingerprint'):stop_owned(previous['fingerprint'])
    current=next(row for row in gpu_inventory() if row['uuid']==gpu['uuid'])
    print(f"Shared GPU: {current['free_mib']} MiB free of {current['total_mib']} MiB before Ollama startup", flush=True)
    asset=next(item for item in read(ROOT/'runtime/ollama-release.json') if item['name']=='ollama-windows-amd64.zip')
    archive=ROOT/'runtime/downloads/ollama-windows-amd64.zip';native=ROOT/'runtime/ollama'
    install=native/'INSTALL.json'
    if not install.exists():
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('need at least 8 GiB for runtime unpacking')
        download(asset['browser_download_url'],archive,asset['digest'].removeprefix('sha256:'),asset['size'])
        safe_extract(archive,native)
        candidates=list(native.rglob('ollama.exe'))
        if len(candidates)!=1:raise RuntimeError('ambiguous Ollama executable')
        atomic_json(install,dict(archive_sha256=asset['digest'],exe=candidates[0].relative_to(native).as_posix(),exe_sha256=sha(candidates[0])))
    metadata=read(install);exe=native/metadata['exe']
    if sha(exe)!=metadata['exe_sha256']:raise RuntimeError('Ollama executable altered')
    models,model_receipt=model_store(extra_models)
    with socket.socket() as available:available.bind(('127.0.0.1',0));port=available.getsockname()[1]
    environment={key:value for key,value in os.environ.items() if not key.startswith('OLLAMA_')}
    environment.update(OLLAMA_HOST=f'127.0.0.1:{port}',OLLAMA_MODELS=str(models.resolve()),
                     OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1',CUDA_VISIBLE_DEVICES=gpu['uuid'])
    stdout=(work/'ollama.stdout.log').open('a',encoding='utf-8');stderr=(work/'ollama.stderr.log').open('a',encoding='utf-8')
    child=subprocess.Popen([str(exe.resolve()),'serve'],cwd=exe.parent,env=environment,stdout=stdout,stderr=stderr,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    fingerprint=process_fingerprint(child.pid)
    atomic_json(old,dict(fingerprint=fingerprint,gpu=gpu,port=port))
    client=StrictOllama(f'http://127.0.0.1:{port}',600)
    try:
        deadline=time.monotonic()+120
        while True:
            if child.poll() is not None:raise RuntimeError('Ollama exited; see stderr log')
            try:
                if client.version()=={'version':'0.32.14'}:break
            except Exception:pass
            if time.monotonic()>deadline:raise RuntimeError('pinned Ollama startup timeout')
            time.sleep(.5)
        descriptions={}
        for item in read(ROOT/'runtime/model_manifests/IDENTITY.json'):
            ref=ModelRef(**item['model']);client.verify_model(ref);description=client.describe(ref)
            if description.get('remote_host') or description.get('remote_model'):raise RuntimeError('cloud model forbidden')
            descriptions[item['role']]=description
        atomic_json(work/'environment.json',dict(gpu=gpu,ollama=client.version(),models=read(ROOT/'runtime/model_manifests/IDENTITY.json'),
            model_setup=model_receipt,descriptions=descriptions,exe_sha256=metadata['exe_sha256']))
        yield client,gpu
    finally:
        if child.poll() is None:child.terminate()
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
        stdout.close();stderr.close()
