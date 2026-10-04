"""Build a GitHub upload ZIP from files eligible for Git.

Run from any directory with Python 3.9 or newer and Git installed. No MATLAB
code is executed. Git history and ignored files are excluded automatically.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'outputs/holographic-quantum-turbulence-github.zip'
MAX_FILE_BYTES = 100 * 1024 * 1024


def main() -> None:
    """Apply the Git/file-size policy, then package only eligible repository files."""
    # Let Git's ignore rules determine the source collection without altering its index.
    result = subprocess.run(
        ['git', '-C', str(ROOT), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        check=True, capture_output=True,
    )
    names = sorted(set(result.stdout.decode('utf-8').split('\0')) - {''})
    manifest = json.loads((ROOT / 'data/metadata/manifest.json').read_text(encoding='utf-8'))
    included_data = {item['path'] for item in manifest['files'] if item['distribution'] == 'git'}
    # Reject generated or oversized data even if they were accidentally added to Git.
    selected = []
    for name in names:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Invalid package path: {name}')
        if name.startswith(('.git/', '.local/', 'outputs/')) and name != 'outputs/README.md':
            raise ValueError(f'Local artifact must not be included: {name}')
        if path.suffix.lower() == '.mat' and name not in included_data:
            raise ValueError(f'Generated data must not be included: {name}')
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'File exceeds the GitHub Git limit: {name}')
        selected.append((name, path))
    # Store the deliverable and its checksum under the ignored output directory.
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DESTINATION, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in selected:
            archive.write(path, 'holographic-quantum-turbulence/' + name)
    checksum = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    DESTINATION.with_suffix('.zip.sha256').write_text(
        f'{checksum}  {DESTINATION.name}\n', encoding='ascii',
    )
    print(f'Packaged {len(selected)} files into {DESTINATION.name}')
    print(f'Archive size: {DESTINATION.stat().st_size:,} bytes')
    print(f'SHA-256: {checksum}')


if __name__ == '__main__':
    main()
