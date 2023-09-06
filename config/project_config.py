"""
Config class implementation: stores the configuration information
"""
import json
import re
from dataclasses import field
from pathlib import Path
from re import Pattern

# pylint: disable=no-name-in-module
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from config.constants import PROJECT_ROOT


@dataclass
class Lab:
    """
    BaseModel for labs
    """
    name: str = field(default_factory=str)
    coverage: int = field(default_factory=int)


@dataclass
class Addon:
    """
    BaseModel for addons
    """
    name: str = field(default_factory=str)
    coverage: int = field(default_factory=int)


@dataclass
class Repository:
    """
    BaseModel for repository
    """
    admins: list = field(default_factory=list)
    pr_name_regex: str = field(default_factory=str)
    pr_name_example: str = field(default_factory=str)


@dataclass
class ProjectConfigDTO:
    """
    BaseModel for ProjectConfig
    """
    labs: list[Lab] = field(default_factory=list[Lab])
    addons: list[Addon] = field(default_factory=list[Addon])
    repository: Repository = field(default_factory=Repository)


class ProjectConfig(ProjectConfigDTO):
    """
    Project Config implementation
    """

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        with config_path.open(encoding='utf-8', mode='r') as config_file:
            json_content = config_file.read()
        # pylint: disable=no-member
        self._dto = ProjectConfigDTO.__pydantic_model__.parse_raw(f"{json_content}")

    def get_thresholds(self) -> dict:
        """
        Returns labs thresholds
        """
        all_thresholds = {}
        labs_thresholds = {lab.name: lab.coverage for lab in self._dto.labs}
        addons_thresholds = {addon.name: addon.coverage for addon in self._dto.addons}
        all_thresholds.update(labs_thresholds)
        all_thresholds.update(addons_thresholds)
        return all_thresholds

    def get_labs_names(self) -> list:
        """
        Returns labs names
        """
        return [lab.name for lab in self._dto.labs]

    def get_labs_paths(self, include_addons: bool = True) -> list:
        """
        Returns labs paths
        """
        labs_list = self.get_labs_names()
        if include_addons:
            labs_list.extend(self.get_addons_names())
        return [PROJECT_ROOT / lab for lab in labs_list]

    def get_addons_names(self) -> list:
        """
        Returns addons names
        """
        return [addon.name for addon in self._dto.addons]

    def get_admins(self) -> list[str]:
        """
        Returns admins names
        """
        return list(self._dto.repository.admins)

    def get_pr_name_regex(self) -> Pattern:
        """
        Return pull request name regex example
        """
        return re.compile(self._dto.repository.pr_name_regex)

    def get_pr_name_example(self) -> str:
        """
        Returns pull request name example
        """
        return str(self._dto.repository.pr_name_example)

    def update_thresholds(self, new_thresholds: dict[str, int]) -> None:
        """
        Returns json content from project_config.json with updated thresholds
        """
        for index, lab in enumerate(self._dto.labs):
            self._dto.labs[index] = \
                Lab(name=lab.name, coverage=new_thresholds.get(lab.name, lab.coverage))
        for index, addon in enumerate(self._dto.addons):
            self._dto.addons[index] = \
                Addon(name=addon.name, coverage=new_thresholds.get(addon.name, addon.coverage))

    def __str__(self) -> str:
        """
        Returns a string with fields
        """
        return f'{self._dto}'

    def get_json(self) -> str:
        """
        Return a json view of ProjectConfig
        """
        return json.dumps(self._dto, indent=4, default=pydantic_encoder)
