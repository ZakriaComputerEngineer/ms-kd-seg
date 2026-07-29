# Paste this into a Kaggle cell to find the dataset. Standalone -- it does not
# need anything else from the notebook to have run.
#
# It prints what is actually mounted, then the exact line to set cfg.data_root to.

import os

print("=" * 70)
for base in ("/kaggle/input", "/kaggle/working"):
    print(f"\n{base}:")
    if not os.path.isdir(base):
        print("   <does not exist>")
        continue
    entries = sorted(os.listdir(base))
    if not entries:
        print("   <empty -- no dataset attached>")
    for name in entries:
        full = os.path.join(base, name)
        kind = "dir " if os.path.isdir(full) else "file"
        link = " (symlink)" if os.path.islink(full) else ""
        print(f"   [{kind}] {name}{link}")
        # One level down is usually enough to recognise the layout.
        if os.path.isdir(full):
            try:
                for sub in sorted(os.listdir(full))[:12]:
                    print(f"            {sub}")
            except OSError as exc:
                print(f"            <unreadable: {exc}>")

# followlinks=True matters: Kaggle mounts attached datasets as symlinks and a
# plain os.walk will not descend into them.
print("\n" + "=" * 70)
print("searching for MS_100_patient_registered ...")
hits = []
for base in ("/kaggle/input", "/kaggle/working"):
    if not os.path.isdir(base):
        continue
    for dirpath, dirnames, _ in os.walk(base, followlinks=True):
        if os.path.relpath(dirpath, base).count(os.sep) > 5:
            dirnames[:] = []
            continue
        if "MS_100_patient_registered" in dirnames:
            hits.append(dirpath)

if hits:
    print("\nFOUND. Set this in the configuration cell and rerun it:\n")
    for h in hits:
        print(f'    data_root="{h}",')
else:
    print("\nNOT FOUND anywhere under /kaggle/input.")
    print("The dataset is not attached to this session. In the right-hand panel click")
    print("'+ Add Input', search for  ms3seg  , and add")
    print("'MS3SEG - MS MRI Tri-Mask Lesion Segmentation'. Then rerun this cell.")
print("=" * 70)
