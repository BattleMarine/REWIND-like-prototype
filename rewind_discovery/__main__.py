"""인증은 notebook에서 주입하며 CLI는 고정 phase 한도 확인만 제공한다."""
import json
from dataclasses import asdict
from .runner import DiscoveryLimits

if __name__=='__main__':print(json.dumps(asdict(DiscoveryLimits()),indent=2))
