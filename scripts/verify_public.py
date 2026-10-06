"""Read-only verification of the sanitized distribution and immutable scientific bytes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 manifest=json.loads((ROOT/'PUBLIC_MANIFEST.json').read_text())
 for name,expected in manifest['sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError(f'Invalid/missing distribution path: {name}')
  if digest(path)!=expected:raise ValueError(f'Distribution hash mismatch: {name}')
 provenance=json.loads((ROOT/'PUBLIC_PROVENANCE.json').read_text())
 for item in provenance['unchanged_scientific_files']:
  if digest(ROOT/item['path'])!=item['sha256']:raise ValueError(f'Scientific evidence changed: {item["path"]}')
 direct=json.loads((ROOT/'evidence-direct-push-v1/summary.json').read_text());cached=json.loads((ROOT/'evidence-cached-adaptation-v1/summary.json').read_text())
 assert direct['trajectories']==288 and not direct['qualified']
 assert cached['completed_fits']==3 and cached['updates_per_seed']==300 and cached['parameters']==115
 assert not cached['positive_signal'] and cached['qualification_remains_failed']
 assert cached['native_steps']==cached['native_resets']==0
 print(f"Distribution verified: {len(manifest['sha256'])} files; {len(provenance['unchanged_scientific_files'])} unchanged scientific/media files. Negative results preserved.")
if __name__=='__main__':main()
