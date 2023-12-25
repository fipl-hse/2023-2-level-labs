"""
Runner for collecting coverage
"""

import json
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe


class CoverageRunError(Exception):
    """
    Error for coverage collection
    """


class CoverageCreateReportError(Exception):
    """
    Error for report creation
    """


def get_target_score(lab_path: Path) -> int:
    """
    Get student's expectations on a final mark
    """
    target_score_file_path = lab_path.joinpath('target_score.txt')
    with open(target_score_file_path, 'r', encoding='utf-8') as target_score_file:
        content = target_score_file.readline()
    return int(content) if content else 0


def extract_percentage_from_report(report_path: Path) -> int:
    """
    Load previous run value
    """
    with report_path.open(encoding='utf-8') as f:
        content = json.load(f)
    return int(content['totals']['percent_covered_display'])


def run_coverage_collection(lab_path: Path, artifacts_path: Path,
                            check_target_score: bool = True) -> int:
    """
    Entrypoint for a single lab coverage collection
    """
    print(f'Processing {lab_path} ...')

    python_exe_path = choose_python_exe()
    mark = ''
    if check_target_score:
        target_score = get_target_score(lab_path)
        mark = f' and mark{target_score}'

    args = [
        '-m', 'coverage', 'run',
        '--include', f'{lab_path.name}/main.py',
        '-m', 'pytest', '-m', f'{lab_path.name}{mark}'
    ]
    res_process = _run_console_tool(str(python_exe_path), args,
                                    debug=True,
                                    cwd=str(lab_path.parent))
    print(res_process.stderr.decode('utf-8') + res_process.stdout.decode('utf-8'))
    report_path = artifacts_path / f'{lab_path.name}.json'
    args = [
        '-m', 'coverage', 'json',
        '-o', str(report_path)
    ]
    res_process = _run_console_tool(str(python_exe_path), args,
                                    debug=True,
                                    cwd=str(lab_path.parent))
    print(res_process.stderr.decode('utf-8') + res_process.stdout.decode('utf-8'))
    return extract_percentage_from_report(report_path)
