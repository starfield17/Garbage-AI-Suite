"""System clock adapter."""

import time

from garbage_deploy.application.ports import ClockPort


class SystemClockAdapter(ClockPort):
    def now(self) -> float:
        return time.time()
