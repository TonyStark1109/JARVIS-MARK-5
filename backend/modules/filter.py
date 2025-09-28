
"""Filter module for extracting code blocks from text."""

import re


def filter_python(txt, *args, **kwargs):  # pylint: disable=unused-argument
    """Extract Python code blocks from text."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, txt, re.DOTALL)

    if matches:
        python_code = matches[0].strip()
        return python_code
    return None


def filter_json(txt, *args, **kwargs):  # pylint: disable=unused-argument
    """Extract JSON code blocks from text."""
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, txt, re.DOTALL)

    if matches:
        json_code = matches[0].strip()
        return json_code
    return None


def filter_code(txt, language, *args, **kwargs):  # pylint: disable=unused-argument
    """Extract code blocks of specified language from text."""
    pattern = rf"```{language}(.*?)```"
    matches = re.findall(pattern, txt, re.DOTALL)

    if matches:
        code = matches[0].strip()
        return code
    return None


def filter_markdown(txt, *args, **kwargs):  # pylint: disable=unused-argument
    """Extract Markdown code blocks from text."""
    pattern = r"```markdown(.*?)```"
    matches = re.findall(pattern, txt, re.DOTALL)

    if matches:
        markdown_code = matches[0].strip()
        return markdown_code
    return None


def filter_html(txt, *args, **kwargs):  # pylint: disable=unused-argument
    """Extract HTML code blocks from text."""
    pattern = r"```html(.*?)```"
    matches = re.findall(pattern, txt, re.DOTALL)

    if matches:
        html_code = matches[0].strip()
        return html_code
    return None


def filter_code_blocks(txt, *args, **kwargs):  # pylint: disable=unused-argument
    """Extract all code blocks from text."""
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, txt, re.DOTALL)

    if matches:
        code_blocks = [match.strip() for match in matches]
        return code_blocks
    return None
