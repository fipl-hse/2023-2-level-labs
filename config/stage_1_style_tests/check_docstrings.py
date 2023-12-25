"""
Check docstrings for conformance to the Google-style-docstrings
"""

import sys
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig


def main(labs_list: list[Path]) -> None:
    """
    Check docstrings.

    Args:
        labs_list (list[Path]): Paths to labs
    """
    all_errors = []
    pydoctest_config = PROJECT_ROOT / 'config' / 'stage_1_style_tests' / 'pydoctest.json'

    for lab_path in labs_list:
        lab_errors = ''
        main_path = lab_path / 'main.py'
        print(f'\nChecking {main_path}')

        pydoctest_args = [
            '--config',
            str(pydoctest_config),
            '--file',
            str(main_path)
        ]
        res_process = _run_console_tool('pydoctest', pydoctest_args, debug=False)
        if res_process.returncode == 0:
            print(f'All docstrings in {main_path} conform to Google-style according to Pydoctest\n')
        else:
            lab_errors += f'Pydoctest errors:\n{res_process.stdout}'

        pydocstyle_args = [
            '-m',
            'pydocstyle',
            str(main_path)
        ]
        res_process = _run_console_tool(str(choose_python_exe()), pydocstyle_args, debug=False)
        if res_process.returncode == 0:
            print(
                f'All docstrings in {main_path} conform to Google-style according to Pydocstyle\n')
        else:
            lab_errors += f'Pydocstyle errors:\n{res_process.stdout}'

        if lab_errors:
            all_errors.append(f'\nDocstrings in {main_path} do not conform to Google-style.\n'
                              f'ERRORS:\n{lab_errors}\n')

    if all_errors:
        print('\n'.join(all_errors))
        print('\nThe docstring check was not successful! Check the logs above.')

        log_file_path = PROJECT_ROOT.joinpath('docstring_check.log')
        with open(file=log_file_path, mode='w', encoding='utf-8') as log_file:
            log_file.write('\n'.join(all_errors))
        print(f'Full check log could be found in: {log_file_path}.\n')

        print('The error explanations for\n'
              'Pydocstyle: http://www.pydocstyle.org/en/stable/error_codes.html')

    sys.exit(bool(all_errors))


if __name__ == '__main__':
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    main(labs_list=project_config.get_labs_paths())
