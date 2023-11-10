"""
Generator for API docs for Sphinx.
"""

import subprocess
from pathlib import Path
from typing import Iterable

from config.constants import PROJECT_CONFIG_PATH
from config.project_config import ProjectConfig


def prepare_args_for_shell(args: Iterable[str]) -> str:
    """
    Util for argument preparation for CLI.

    Args:
        args (list): arguments to join

    Returns:
         str: arguments for CLI
    """
    return ' '.join(args)


def generate_api_docs(labs_paths: list[Path],
                      apidoc_templates_path: Path,
                      overwrite: bool = False) -> None:
    """
    Generate API docs for all laboratory works.

    Iterate over the specified lab* folders under the source_code_root and
    generate the API .rst document in the lab folder, i.e.,
    source_code_root/lab_name/lab_name.api.rst.

    Args:
        labs_paths:
        apidoc_templates_path:
        overwrite:
    """

    for lab_path in labs_paths:
        lab_api_doc_path = lab_path

        args = [
            'sphinx-apidoc',
            '-o',
            lab_api_doc_path,
            '--no-toc',
            '--no-headings',
            '--suffix',
            'api.rst',
            '-t',
            apidoc_templates_path,
            lab_path
        ]
        if overwrite:
            args.insert(-1, '-f')

        excluded_paths = (lab_path.joinpath('tests'),
                          lab_path.joinpath('assets'),
                          lab_path.joinpath('start.py'),
                          lab_path.joinpath('helpers.py'))
        args.extend(excluded_paths)

        command = prepare_args_for_shell(map(str, args))

        print(f'FULL COMMAND: {command}')
        result = subprocess.run(args=command,
                                shell=True,
                                check=True)
        if result.returncode == 0:
            print(f'API DOC FOR {lab_path} GENERATED IN {lab_api_doc_path}\n')
        else:
            print(f'ERROR CODE: {result.returncode!r}. ERROR: {result.stderr!r}\n')


if __name__ == '__main__':
    project_config = ProjectConfig(config_path=PROJECT_CONFIG_PATH)

    templates_path = Path(__file__).parent.joinpath('templates').joinpath('apidoc')

    generate_api_docs(labs_paths=project_config.get_labs_paths(),
                      apidoc_templates_path=templates_path,
                      overwrite=True)
