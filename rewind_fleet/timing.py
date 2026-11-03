"""명시 survey 전용 observe-only 시간 정책. 요청 timeout은5초로 유지."""
from dataclasses import dataclass,asdict
from rewind_archive.contracts import Limits,Budget as TransferBudget
from rewind_archive_index.source import IndexLimits,Budget as IndexBudget,IndexFailure
from rewind_archive_index.drive import DriveLimits

@dataclass(frozen=True)
class ObservedIndexLimits(IndexLimits):
    max_seconds:None=None
    def __post_init__(self):
        if self.max_seconds is not None:raise IndexFailure('fleet_observe_only_seconds')
        IndexLimits(**{**asdict(self),'max_seconds':1}) # 나머지 기존 수치 검증만 재사용.

class ObservedIndexBudget(IndexBudget):
    def check(self):
        rss=self.process.memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
        if rss>self.limits.max_rss_bytes:raise IndexFailure('rss_limit')

@dataclass(frozen=True)
class ObservedTransferLimits(Limits):
    max_elapsed_seconds:None=None
    def __post_init__(self):
        if self.max_elapsed_seconds is not None:raise IndexFailure('fleet_observe_only_seconds')
        Limits(**{**asdict(self),'max_elapsed_seconds':self.request_timeout_seconds})

class ObservedTransferBudget(TransferBudget):
    def check_time(self):
        # 기존 IO API가 반환값을 요청 timeout으로 사용한다. 전체 남은 시간이
        # 아니라 매 요청의5초 horizon이며 started/실제 elapsed는 계속 관측한다.
        return self.limits.request_timeout_seconds

@dataclass(frozen=True)
class ObservedDriveLimits(DriveLimits):
    max_seconds:None=None
    def transport_limits(self,seconds=None):
        if seconds is not None or self.max_seconds is not None:raise IndexFailure('fleet_observe_only_seconds')
        return ObservedTransferLimits(max_transfer_bytes=self.max_body_bytes,max_ranges=self.max_requests,
            request_timeout_seconds=5,media_range_bytes=self.range_bytes)
    def new_transfer_budget(self):return ObservedTransferBudget(self.transport_limits())
