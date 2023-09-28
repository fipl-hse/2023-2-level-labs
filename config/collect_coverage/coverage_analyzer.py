"""
Runner for collecting coverage
"""

import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional

from config.collect_coverage.run_coverage import (CoverageCreateReportError, CoverageRunError,
                                                  run_coverage_collection)
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig

CoverageResults = Mapping[str, Optional[int]]


def collect_coverage(all_labs_names: Iterable[Path],
                     artifacts_path: Path) -> CoverageResults:
    """
    Entrypoint for coverage collection for every required folder
    """
    all_labs_results = {}
    for lab_path in all_labs_names:
        percentage = None
        try:
            if lab_path.name == 'core_utils':
                check_target = False
            else:
                check_target = True
            percentage = run_coverage_collection(lab_path=lab_path, artifacts_path=artifacts_path,
                                                 check_target_score=check_target)
        except (CoverageRunError, CoverageCreateReportError) as e:
            print(e)
        finally:
            all_labs_results[lab_path.name] = percentage
    return all_labs_results


def is_decrease_present(all_labs_results: CoverageResults,
                        previous_coverage_results: dict) -> tuple[bool, dict]:
    """
    Analysis of coverage report versus previous runs
    """
    print('\n\n' + '------' * 3)
    print('REPORT')
    print('------' * 3)
    any_degradation = False
    labs_with_thresholds = {}
    for lab_name, current_lab_percentage in all_labs_results.items():
        prev_lab_percentage = previous_coverage_results.get(lab_name, 0)
        if current_lab_percentage is None:
            current_lab_percentage = 0
        diff = current_lab_percentage - prev_lab_percentage

        print(f'{lab_name:<30}: {current_lab_percentage}% ({"+" if diff >= 0 else ""}{diff})')
        labs_with_thresholds[lab_name] = current_lab_percentage
        if diff < 0:
            any_degradation = True
    print('\n\n' + '------' * 3)
    print('END OF REPORT')
    print('------' * 3 + '\n\n')

    return any_degradation, labs_with_thresholds


def main() -> None:
    """
    Entrypoint for coverage collection
    """
    artifacts_path = PROJECT_ROOT / 'build' / 'coverage'
    artifacts_path.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    coverage_thresholds = project_config.get_thresholds()

    all_labs_names = project_config.get_labs_paths()

    all_labs_results = collect_coverage(all_labs_names, artifacts_path)

    any_degradation, labs_with_thresholds = \
        is_decrease_present(all_labs_results, coverage_thresholds)

    if any_degradation:
        print('Some of labs have worse coverage. We cannot accept this. Write more tests!')
        print('You can copy-paste the following content to the ./config/project_config.json '
              'to update thresholds. \n\n')

        project_config.update_thresholds(labs_with_thresholds)

        print(project_config.get_json())
        sys.exit(1)

    print('Nice coverage. Anyway, write more tests!', end='\n\n')


if __name__ == '__main__':
    main()
