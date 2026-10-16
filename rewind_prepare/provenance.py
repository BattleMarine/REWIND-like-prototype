"""Source ZIP revision 또는 로컬 Git revision과 dirty 범위를 기록한다."""
import re
import subprocess


def revision_identity(root):
    stamp=root/'SOURCE_REVISION.txt'
    revision='unversioned';kind='code_hash_only';dirty=None
    if stamp.exists():
        revision=stamp.read_text(encoding='utf-8').strip()
        if not re.fullmatch(r'[0-9a-fA-F]{40}',revision):raise ValueError('source_revision_stamp_invalid')
        kind='source_revision_stamp'
    elif (root/'.git').exists():
        try:
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            kind='git_head'
        except (OSError,subprocess.CalledProcessError):pass
    if (root/'.git').exists():
        try:dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip())
        except (OSError,subprocess.CalledProcessError):pass
    return {'revision':revision,'revision_kind':kind,'dirty':dirty,'dirty_scope':'entire_git_checkout_when_available',
            'code_hash_scope':['rewind_prepare/*.py','rewind_data/*.py','rewind_archive/*.py','preparation_spec_bytes']}
