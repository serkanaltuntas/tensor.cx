"""Pytest entry point that rejects skipped, XFAIL and non-strict XPASS reports."""
import sys

import pytest


class RequiredTests:
    def __init__(self):
        self.rejected = False

    def pytest_runtest_logreport(self, report):
        if report.skipped or hasattr(report, 'wasxfail'):
            self.rejected = True

    def pytest_sessionfinish(self, session, exitstatus):
        if self.rejected:
            session.exitstatus = 1


if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:], plugins=[RequiredTests()]))
