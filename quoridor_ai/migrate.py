import zipfile,io,torch,json
from pathlib import Path
from .safe_loader import load_checkpoint
def inspect_and_extract(zip_path,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);report=[]
 with zipfile.ZipFile(zip_path) as z:
  for n in z.namelist():
   if not n.endswith('latest.pt'):continue
   raw=z.read(n);d=load_checkpoint(io.BytesIO(raw),map_location='cpu');seed=n.split('/')[-2];dest=out/f'{seed}_legacy.pt';dest.write_bytes(raw);report.append({'seed':seed,'iteration':d.get('iteration'),'config':d.get('config'),'replay':len(d.get('replay',[])),'file':str(dest.name)})
 (out/'migration_report.json').write_text(json.dumps(report,indent=2));return report
