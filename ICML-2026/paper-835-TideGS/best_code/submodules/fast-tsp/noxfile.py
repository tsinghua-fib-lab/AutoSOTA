from __future__ import annotations

import nox

nox.options.sessions = ['lint', 'tests']


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install('pre-commit')
    session.run('pre-commit', 'run', '--all-files', *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install('.[test]')
    session.run('pytest', *session.posargs)
