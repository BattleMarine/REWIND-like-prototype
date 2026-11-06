"""Compare official sequence folder paths with Drive without reading file payloads."""
import collections
import time
from zipfile import ZipFile

from unrealego2_upload_drive import CATEGORIES, FOLDER, DriveSafetyError


def folder_survey(service, root_id, split_zip, category_envs, *, max_requests, max_seconds):
    """Walk only prefixes of the selected split paths, listing up to 24 parents per query."""
    started = time.monotonic()
    selected = set(category_envs)
    if not selected or any(cat not in CATEGORIES for cat, _ in selected):
        raise ValueError("select one or more RGB/pose environments")
    expected = {}
    with ZipFile(split_zip) as archive:
        for split in ("train", "validation", "test"):
            for sequence in archive.read(f"UnrealEgoData2/{split}.txt").decode("utf-8-sig").splitlines():
                sequence = sequence.strip().rstrip("/")
                parts = sequence.split("/")
                if len(parts) != 6 or any(not p or p in (".", "..") for p in parts):
                    raise ValueError("unexpected official sequence path")
                for kind in ("rgb", "pose"):
                    cat = "UnrealEgoData2_" + ("test" if split == "test" else "train_val") + "_" + kind
                    if (cat, parts[0]) in selected:
                        key = (cat, *parts)
                        if key in expected:
                            raise ValueError("overlapping official split")
                        expected[key] = split
    if not expected:
        raise ValueError("official split selection is empty")
    parents = {(): root_id}
    unresolved = {}
    calls = 0

    def literal(s):
        return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"

    for level in range(7):  # category / environment / light / human / glasses / category / motion
        needed = collections.defaultdict(set)
        for key in expected:
            prefix = key[:level]
            if prefix in parents:
                needed[prefix].add(key[level])
        fresh = {}
        paths = sorted(needed)
        for start in range(0, len(paths), 24):
            group = paths[start:start + 24]
            ids = {parents[p]: p for p in group}
            query = "trashed = false and (" + " or ".join(literal(i) + " in parents" for i in ids) + ")"
            if level == 0:
                query += " and (" + " or ".join("name = " + literal(n) for n in sorted(needed[()])) + ")"
            found = collections.defaultdict(list)
            token, tokens = None, set()
            while True:
                if calls >= max_requests or time.monotonic() - started >= max_seconds:
                    raise DriveSafetyError("folder survey limit")
                calls += 1
                page = service.files().list(q=query, fields="nextPageToken,incompleteSearch,files(id,name,parents,mimeType)",
                    pageSize=1000, pageToken=token, includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
                if page.get("incompleteSearch"):
                    raise DriveSafetyError("incomplete folder search")
                for item in page.get("files", []):
                    for parent in item.get("parents", []):
                        prefix = ids.get(parent)
                        if prefix is not None and item["name"] in needed[prefix]:
                            found[prefix + (item["name"],)].append(item)
                token = page.get("nextPageToken")
                if not token:
                    break
                if token in tokens:
                    raise DriveSafetyError("repeated folder listing page")
                tokens.add(token)
            for p in group:
                for name in needed[p]:
                    path = p + (name,)
                    items = found[path]
                    if not items:
                        unresolved[path] = "folder_absent"
                    elif len(items) == 1 and items[0]["mimeType"] == FOLDER:
                        fresh[path] = items[0]["id"]
                    else:
                        unresolved[path] = "ambiguous_or_wrong_type"
        parents = fresh
    result = []
    for key, split in sorted(expected.items()):
        issue = next((key[:i] for i in range(1, 8) if key[:i] in unresolved), None)
        status = unresolved[issue] if issue else "folder_present"
        result.append({"category": key[0], "sequence": "/".join(key[1:]), "split": split,
                       "status": status, "first_problem": "/".join(issue) if issue else None})
    return {"scope": "official sequence folders only", "requests": calls,
            "elapsed_seconds": time.monotonic() - started,
            "counts": dict(collections.Counter(r["status"] for r in result)), "sequences": result}
