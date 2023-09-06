"""
Checks dependencies
"""

import re
import sys

from config.constants import PROJECT_ROOT


def get_requirements() -> list:
    """
    Returns a list of dependencies
    """
    with (PROJECT_ROOT / 'requirements.txt').open(encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def compile_pattern() -> re.Pattern:
    """
    Returns the compiled pattern
    """
    return re.compile(r'\w+(-\w+)*==\d(\.\d+)+')


def check_dependencies(lines: list, compiled_pattern: re.Pattern):
    """
    Checks that dependencies confirm to the template
    """
    if sorted(lines) != lines:
        print('Dependencies in requirements.txt do not conform to the template.')
        return False
    for line in lines:
        if not re.search(compiled_pattern, line):
            print('Dependencies in requirements.txt do not conform to the template.')
            return False
    print('Dependencies in requirements.txt: OK.')
    return True


def main():
    """
    Calls functions
    """
    lines = get_requirements()
    compiled_pattern = compile_pattern()

    sys.exit(not check_dependencies(lines, compiled_pattern))


if __name__ == '__main__':
    main()
