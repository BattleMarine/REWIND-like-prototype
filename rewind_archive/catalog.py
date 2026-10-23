"""상한 있는 sequence 표와 관측 행. payload 검증이나 seek index가 아니다."""

import re
from .contracts import canonical, digest
from .cache_policy import WARM_POLICIES

CATALOG_BYTES=6*1024**2
MAX_SEQUENCES=128
MAX_ENTRIES=16000


class Catalog:
    def __init__(self,max_sequences=MAX_SEQUENCES,max_entries=MAX_ENTRIES,policy=None):
        self.max_sequences=min(max_sequences,256 if policy in WARM_POLICIES else MAX_SEQUENCES)
        self.max_entries=min(max_entries,MAX_ENTRIES)
        self.sequences=[];self.entries=[];self.observed=[]
        self.truncated=False;self.unclassified=0;self.last_unclassified=None

    def add(self,observation):
        match=re.fullmatch(r"(.+)/(?:json/frame_(0|[1-9][0-9]{0,9})\.json|fisheye_final_image/(camera_left|camera_right)/final_(0|[1-9][0-9]{0,9})\.png)",observation["member"])
        if match is None:
            self.unclassified+=1;self.last_unclassified=observation["member"];return
        sequence=match[1];frame=int(match[2] or match[4])
        if sequence not in self.sequences:
            if len(self.sequences)>=self.max_sequences:self.truncated=True;return
            self.sequences.append(sequence)
        if len(self.entries)>=self.max_entries:self.truncated=True;return
        self.entries.append({"sequence_ref":self.sequences.index(sequence),"frame":frame,
            "kind":"rgb" if match[3] else "pose","side":match[3]})
        self.observed.append(observation)

    def document(self,usable):
        trust="bounded_observation_only" if usable else "untrusted_failed_source_or_parser"
        rows=[{**entry,**{key:obs[key] for key in ("header_ordinal","regular_ordinal","size_bytes",
            "payload_complete","payload_bytes_returned","stored")},"trust":trust}
            for entry,obs in zip(self.entries,self.observed)]
        body={"schema_version":"archive_catalog_v1","sequences":self.sequences,"entries":rows,
            "max_sequences":self.max_sequences,"max_entries":self.max_entries,"truncated":self.truncated,
            "unclassified_regular_members":self.unclassified,"last_unclassified_member":self.last_unclassified,
            "trust":trust,"full_source_listing_claimed":False,"seek_index":False}
        return {**body,"content_sha256":digest(body)}

    def save(self,output,usable):
        from .contracts import PreparationError
        doc=self.document(usable);raw=canonical(doc)+b"\n"
        if len(raw)>CATALOG_BYTES:raise PreparationError("catalog_bytes_limit")
        (output/"observation_catalog.json").write_bytes(raw)
        return {"relative_path":"observation_catalog.json","content_sha256":doc["content_sha256"],
            "bytes":len(raw),"entries":len(doc["entries"]),"sequences":len(doc["sequences"]),
            "truncated":doc["truncated"],"unclassified_regular_members":self.unclassified,"trust":doc["trust"],
            "full_source_listing_claimed":False,"seek_index":False}
