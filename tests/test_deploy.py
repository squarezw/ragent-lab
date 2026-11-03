"""Exercise the real deployment script against controlled Docker/curl commands."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'deploy/deploy.sh'
REVISION = 'a' * 40


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.release = self.root / 'releases' / REVISION
        (self.release / 'deploy').mkdir(parents=True)
        shutil.copy(SCRIPT, self.release / 'deploy/deploy.sh')
        (self.release / 'release.env').write_text('API_IMAGE=test-api\nWEB_IMAGE=test-web\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.log = self.root / 'operations'
        self.stub('flock', 'exit 0')
        self.stub('sleep', 'exit 0')
        self.stub('docker', '''printf '%s\\n' "$*" >> "$TEST_LOG"
if [[ "$1" == inspect ]]; then printf '%s\\n' "${LEGACY_RUNNING:-false}"; fi
if [[ "${FAIL_PULL:-0}" == 1 && "$*" == *" pull" ]]; then exit 1; fi
if [[ "${FAIL_UP:-0}" == 1 && "$*" == *"$TEST_REVISION"* && "$*" == *" up "* ]]; then exit 1; fi
''')
        self.stub('curl', 'printf \'{"revision":"%s","title":"Ragent Lab"}\\n\' "$TEST_REVISION"')
        self.env = dict(os.environ, PATH=f'{self.bin}:{os.environ["PATH"]}', LAB_DEPLOY_ROOT=str(self.root), TEST_LOG=str(self.log), TEST_REVISION=REVISION)

    def stub(self, name, body):
        path = self.bin / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)

    def run_release(self, **env):
        return subprocess.run(['bash', str(self.release / 'deploy/deploy.sh'), REVISION], env=dict(self.env, **env), capture_output=True, text=True)

    def test_success_promotes_release_after_checks(self):
        result = self.run_release(LEGACY_RUNNING='true')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'current').resolve(), self.release)
        log = self.log.read_text()
        self.assertLess(log.index(' pull'), log.index('stop rag-lab'))

    def test_failed_upgrade_restores_previous_release(self):
        previous = self.root / 'releases' / ('b' * 40)
        previous.mkdir()
        (previous / 'release.env').write_text('API_IMAGE=previous\n')
        (self.root / 'current').symlink_to(previous)
        result = self.run_release(FAIL_UP='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / 'current').resolve(), previous)
        log = self.log.read_text()
        self.assertIn(' down ', log)
        self.assertIn(f'--env-file {previous}/release.env', log)

    def test_failed_first_migration_restarts_legacy(self):
        result = self.run_release(FAIL_UP='1', LEGACY_RUNNING='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('start rag-lab', self.log.read_text())
        self.assertFalse((self.root / 'current').exists())

    def test_pull_failure_does_not_stop_live_service(self):
        result = self.run_release(FAIL_PULL='1', LEGACY_RUNNING='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('stop rag-lab', self.log.read_text())
