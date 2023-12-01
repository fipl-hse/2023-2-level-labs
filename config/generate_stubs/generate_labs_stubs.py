"""
Generator of all labs
"""

from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.generate_stubs.generator import cleanup_code
from config.generate_stubs.run_generator import format_stub_file, sort_stub_imports
from config.project_config import ProjectConfig


def generate_all_stubs(project_config: ProjectConfig) -> None:
    """
    Generates stubs for all labs
    """
    labs = project_config.get_labs_names()
    for lab_name in labs:
        print(f'Generating stubs for {lab_name}')
        source_code = cleanup_code(PROJECT_ROOT / lab_name / 'main.py')
        with (PROJECT_ROOT / lab_name / 'main_stub.py').open(mode='w',
                                                             encoding='utf-8') as f:
            f.write(source_code)
        format_stub_file(PROJECT_ROOT / lab_name / 'main_stub.py')
        sort_stub_imports(PROJECT_ROOT / lab_name / 'main_stub.py')

        source_code = cleanup_code(PROJECT_ROOT / lab_name / 'start.py')
        with (PROJECT_ROOT / lab_name / 'start_stub.py').open(mode='w',
                                                              encoding='utf-8') as f:
            f.write(source_code)
        format_stub_file(PROJECT_ROOT / lab_name / 'start_stub.py')
        sort_stub_imports(PROJECT_ROOT / lab_name / 'start_stub.py')


def main() -> None:
    """
    Entrypoint for stub generation
    """
    proj_conf = ProjectConfig(PROJECT_CONFIG_PATH)
    generate_all_stubs(proj_conf)


if __name__ == '__main__':
    main()
