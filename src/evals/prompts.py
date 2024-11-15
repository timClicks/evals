r"""Grab the YAML headers from all the SMD Prompts."""

import argparse
import csv
import re
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML

TABLES_DIR = Path(__file__).parent.parent.parent / "tables"
PROMPT_PATH = TABLES_DIR / "prompt.csv"

# Extract the Yaml header from a file
HEADER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class YAMLHeader(BaseModel):
    r"""Model for the YAML headers in the SMD files.

    They look like this:
    ---
    type: Prompt
    id: stencila/create/paragraph
    version: "0.1.0"
    name: Create Paragraph Prompt
    description: Create a paragraph.
    keywords: paragraph
    instruction-type: Create
    instruction-pattern: (?i)\b(para(graph)?)\b
    node-type: Paragraph
    ---

    """

    type: str = Field(alias="instruction-type")
    name: str = Field(alias="id")
    version: str
    created_by: str = "stencila"
    # description: str = ""
    # keywords: str
    # instruction_type: str = Field(..., alias="instruction-type")
    # instruction_pattern: str = Field(..., alias="instruction-pattern")
    # node_type: str = Field(..., alias="node-type")
    # model_config = {
    #     "populate_by_name": True,
    #     "extra": "ignore",
    # }


def parse_yaml_headers(folder: Path) -> list[YAMLHeader]:
    yaml = YAML(typ="safe")
    headers = []

    for file_path in folder.rglob("*.smd"):
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text()
            if match := HEADER_PATTERN.match(content):
                yaml_data = yaml.load(match.group(1))
                try:
                    header = YAMLHeader.model_validate(yaml_data)
                    headers.append(header)
                except ValidationError:
                    logger.warning(f"not a prompt {file_path}:")
                    continue
                logger.info(f"parsed {file_path}")

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

    return headers


def write_to_csv(headers: list[YAMLHeader], output_file: Path):
    with output_file.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(YAMLHeader.model_fields.keys())  # Write the header row
        for header in headers:
            writer.writerow(header.model_dump().values())  # Write each header's values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing files to parse")
    # parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()

    headers = parse_yaml_headers(Path(args.folder))
    write_to_csv(headers, PROMPT_PATH)


if __name__ == "__main__":
    main()
