"""
Check docstrings for conformance to the Google-style-docstrings
"""
import subprocess
import sys
from pathlib import Path

from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig


def prepare_args_for_shell(args: list[object]) -> str:
    """
    Returns arguments for shell
    """
    return " ".join(map(str, args))


def main(labs_list: list[Path]) -> None:
    """Summary

    Description

    Args:
        labs_list:

    Returns:

    """

    all_errors = []

    for lab_path in labs_list:
        lab_errors = ''
        main_path = lab_path / 'main.py'

        print(f'\nChecking {main_path}')
        darglint_args_list = [
            'darglint',
            '--docstring-style',
            'google',
            '--strictness',
            'full',
            '--enable',
            'DAR104',
            main_path
        ]
        darglint_args = prepare_args_for_shell(darglint_args_list)
        print(f'FULL DARGLINT COMMAND: {darglint_args}')

        result = subprocess.run(args=darglint_args,  # pylint: disable=subprocess-run-check
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=True)
        if result.returncode == 0:
            print(f'All docstrings in {main_path} conform to Google-style according to Darglint\n')
        else:
            lab_errors += f'Darglint errors:\n{result.stdout}'

        pydocstyle_args_list = [
            'python',
            '-m',
            'pydocstyle',
            main_path
        ]
        pydocstyle_args = prepare_args_for_shell(pydocstyle_args_list)
        print(f'FULL PYDOCSTYLE COMMAND: {pydocstyle_args}')

        result = subprocess.run(args=pydocstyle_args,  # pylint: disable=subprocess-run-check
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=True)
        if result.returncode == 0:
            print(
                f'All docstrings in {main_path} conform to Google-style according to Pydocstyle\n')
        else:
            lab_errors += f'Pydocstyle errors:\n{result.stdout}'

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
              'Darglint: https://github.com/terrencepreilly/darglint#error-codes\n'
              'Pydocstyle: http://www.pydocstyle.org/en/stable/error_codes.html')

    sys.exit(bool(all_errors))


if __name__ == '__main__':
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    main(labs_list=project_config.get_labs_paths())
