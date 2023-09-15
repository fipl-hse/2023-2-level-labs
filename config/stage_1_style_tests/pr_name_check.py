# pylint: skip-file
import argparse
import re
import sys
from re import Pattern

from config.constants import PROJECT_CONFIG_PATH
from config.project_config import ProjectConfig


def convert_raw_pr_name(pr_name_raw: str) -> str:
    return pr_name_raw.replace('_', ' ')


def is_matching_name(pr_name: str, compiled_pattern: Pattern, example_name: str) -> bool:
    if not re.search(compiled_pattern, pr_name):
        print('Your Pull Request title does not confirm to the template.')
        print(example_name, end='\n\n')
        return False

    print('Your Pull Request name confirms to provided template.')
    return True


def is_author_admin(author_login: str, project_config: ProjectConfig) -> bool:
    admins_logins = project_config.get_admins()

    return author_login in admins_logins


if __name__ == '__main__':
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)

    parser = argparse.ArgumentParser(description='Checks that PR name is done using the template')
    parser.add_argument('--pr-name', type=str, required=True, help='Current PR name')
    parser.add_argument('--pr-author', type=str, required=True, help='Current PR author')
    args: argparse.Namespace = parser.parse_args()

    if '[skip-name]' in args.pr_name:
        print("Skipping PR name checks.")
        sys.exit(0)

    if is_author_admin(args.pr_author, project_config):
        print('Skipping PR name checks due to author.')
        sys.exit(0)

    pr_name = convert_raw_pr_name(args.pr_name)

    compiled_pattern = project_config.get_pr_name_regex()
    example_name = project_config.get_pr_name_example()

    sys.exit(not is_matching_name(pr_name, compiled_pattern, example_name))
