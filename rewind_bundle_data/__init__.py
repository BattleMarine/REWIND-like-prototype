"""완료 bundle의 학습 차단 T=1 데이터 공급. 모델·UI·optimizer 없음."""
from .reader import PreparedBundleReader, ReaderLimits

__all__ = ['PreparedBundleReader', 'ReaderLimits']
