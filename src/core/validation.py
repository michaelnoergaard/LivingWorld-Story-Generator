"""Input validation utilities for LivingWorld.

This module provides comprehensive input validation for all user inputs,
API parameters, file operations, and database queries.
"""

import re
import os
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from datetime import datetime

from typing import Any


class ValidationError(Exception):
    """Custom validation error with detailed error information."""

    def __init__(self, field: str, value: Any, message: str, constraint: str = None):
        """
        Initialize validation error.

        Args:
            field: Field name that failed validation
            value: Invalid value that was provided
            message: Error message explaining what went wrong
            constraint: Optional constraint that was violated
        """
        self.field = field
        self.value = value
        self.message = message
        self.constraint = constraint
        super().__init__(f"Validation failed for field '{field}': {message}")


def validate_id(id_value: Union[int, str], field_name: str = "id", min_value: int = 1) -> int:
    """
    Validate that an ID is a positive integer.

    Args:
        id_value: Value to validate
        field_name: Name of the field being validated
        min_value: Minimum allowed value (default: 1)

    Returns:
        Validated integer ID

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(id_value, str):
        try:
            id_value = int(id_value)
        except ValueError:
            raise ValidationError(
                field=field_name,
                value=id_value,
                message="ID must be a positive integer",
                constraint=f"integer >= {min_value}"
            )

    if not isinstance(id_value, int):
        raise ValidationError(
            field=field_name,
            value=id_value,
            message="ID must be an integer",
            constraint=f"integer >= {min_value}"
        )

    if id_value < min_value:
        raise ValidationError(
            field=field_name,
            value=id_value,
            message=f"ID must be at least {min_value}",
            constraint=f"integer >= {min_value}"
        )

    return id_value


def validate_string(
    value: str,
    field_name: str,
    min_length: int = 0,
    max_length: Optional[int] = None,
    required: bool = True,
    strip_whitespace: bool = True,
    pattern: Optional[str] = None,
    allowed_chars: Optional[str] = None,
) -> str:
    """
    Validate a string input.

    Args:
        value: String value to validate
        field_name: Name of the field being validated
        min_length: Minimum required length
        max_length: Maximum allowed length
        required: Whether the field is required
        strip_whitespace: Whether to strip whitespace
        pattern: Regular pattern the string must match
        allowed_chars: String of allowed characters

    Returns:
        Validated string

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(
                field=field_name,
                value=value,
                message="Field is required",
                constraint=f"string, min_length={min_length}"
            )
        return ""

    if not isinstance(value, str):
        raise ValidationError(
            field=field_name,
            value=value,
            message="Field must be a string",
            constraint=f"string, min_length={min_length}"
        )

    # Strip whitespace if requested
    if strip_whitespace:
        value = value.strip()

    # Check length constraints
    length = len(value)
    if length < min_length:
        raise ValidationError(
            field=field_name,
            value=value,
            message=f"Field must be at least {min_length} characters long",
            constraint=f"string, min_length={min_length}"
        )

    if max_length is not None and length > max_length:
        raise ValidationError(
            field=field_name,
            value=value,
            message=f"Field must be no more than {max_length} characters long",
            constraint=f"string, max_length={max_length}"
        )

    # Check pattern constraint
    if pattern:
        if not re.match(pattern, value, re.DOTALL):
            raise ValidationError(
                field=field_name,
                value=value,
                message="Field contains invalid characters or format",
                constraint=f"pattern: {pattern}"
            )

    # Check allowed characters
    if allowed_chars:
        invalid_chars = set(value) - set(allowed_chars)
        if invalid_chars:
            raise ValidationError(
                field=field_name,
                value=value,
                message=f"Field contains invalid characters: {', '.join(invalid_chars)}",
                constraint=f"allowed_chars: {allowed_chars}"
            )

    return value


def validate_choice(
    choice: int,
    field_name: str = "choice",
    min_choice: int = 1,
    max_choice: int = 3,
) -> int:
    """
    Validate a choice number.

    Args:
        choice: Choice number to validate
        field_name: Name of the field being validated
        min_choice: Minimum allowed choice
        max_choice: Maximum allowed choice

    Returns:
        Validated choice number

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(choice, int):
        try:
            choice = int(choice)
        except ValueError:
            raise ValidationError(
                field=field_name,
                value=choice,
                message="Choice must be a number",
                constraint=f"integer between {min_choice} and {max_choice}"
            )

    if choice < min_choice or choice > max_choice:
        raise ValidationError(
            field=field_name,
            value=choice,
            message=f"Choice must be between {min_choice} and {max_choice}",
            constraint=f"integer between {min_choice} and {max_choice}"
        )

    return choice


def validate_file_path(
    file_path: Union[str, Path],
    field_name: str = "file_path",
    must_exist: bool = True,
    must_be_file: bool = True,
    allowed_extensions: Optional[List[str]] = None,
    parent_dir: Optional[Path] = None,
) -> Path:
    """
    Validate a file path.

    Args:
        file_path: Path to validate
        field_name: Name of the field being validated
        must_exist: Whether the file must exist
        must_be_file: Whether the path must point to a file
        allowed_extensions: List of allowed file extensions
        parent_dir: Optional parent directory to validate against

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not isinstance(file_path, Path):
        raise ValidationError(
            field=field_name,
            value=file_path,
            message="File path must be a string or Path object",
            constraint="Path object or string"
        )

    # Normalize path
    file_path = file_path.resolve()

    # Check if parent directory is specified and valid
    if parent_dir:
        parent_dir = Path(parent_dir).resolve()
        try:
            file_path.relative_to(parent_dir)
        except ValueError:
            raise ValidationError(
                field=field_name,
                value=str(file_path),
                message=f"File path must be within parent directory: {parent_dir}",
                constraint=f"subpath of {parent_dir}"
            )

    # Check if file exists
    if must_exist:
        if not file_path.exists():
            raise ValidationError(
                field=field_name,
                value=str(file_path),
                message="File does not exist",
                constraint="existing file path"
            )

        # Check if it's a file or directory
        if must_be_file and not file_path.is_file():
            raise ValidationError(
                field=field_name,
                value=str(file_path),
                message="Path must point to a file, not a directory",
                constraint="file path"
            )

    # Check file extension
    if allowed_extensions:
        if file_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                field=field_name,
                value=str(file_path),
                message=f"File extension must be one of: {', '.join(allowed_extensions)}",
                constraint=f"extensions: {allowed_extensions}"
            )

    return file_path


def validate_directory_path(
    dir_path: Union[str, Path],
    field_name: str = "directory_path",
    must_exist: bool = True,
    must_be_dir: bool = True,
    parent_dir: Optional[Path] = None,
    create_if_missing: bool = False,
) -> Path:
    """
    Validate a directory path.

    Args:
        dir_path: Directory path to validate
        field_name: Name of the field being validated
        must_exist: Whether the directory must exist
        must_be_dir: Whether the path must point to a directory
        parent_dir: Optional parent directory to validate against
        create_if_missing: Whether to create directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if not isinstance(dir_path, Path):
        raise ValidationError(
            field=field_name,
            value=dir_path,
            message="Directory path must be a string or Path object",
            constraint="Path object or string"
        )

    # Normalize path
    dir_path = dir_path.resolve()

    # Check if parent directory is specified and valid
    if parent_dir:
        parent_dir = Path(parent_dir).resolve()
        try:
            dir_path.relative_to(parent_dir)
        except ValueError:
            raise ValidationError(
                field=field_name,
                value=str(dir_path),
                message=f"Directory path must be within parent directory: {parent_dir}",
                constraint=f"subpath of {parent_dir}"
            )

    # Check if directory exists
    if must_exist:
        if not dir_path.exists():
            if create_if_missing:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValidationError(
                        field=field_name,
                        value=str(dir_path),
                        message=f"Cannot create directory: {e}",
                        constraint="creatable directory path"
                    )
            else:
                raise ValidationError(
                    field=field_name,
                    value=str(dir_path),
                    message="Directory does not exist",
                    constraint="existing directory path"
                )

        # Check if it's a directory
        if must_be_dir and not dir_path.is_dir():
            raise ValidationError(
                field=field_name,
                value=str(dir_path),
                message="Path must point to a directory, not a file",
                constraint="directory path"
            )

    return dir_path


def validate_json_string(
    json_str: str,
    field_name: str = "json_data",
) -> Dict[str, Any]:
    """
    Validate and parse a JSON string.

    Args:
        json_str: JSON string to validate and parse
        field_name: Name of the field being validated

    Returns:
        Parsed JSON data as dictionary

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(json_str, str):
        raise ValidationError(
            field=field_name,
            value=json_str,
            message="JSON data must be a string",
            constraint="valid JSON string"
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(
            field=field_name,
            value=json_str,
            message=f"Invalid JSON format: {e}",
            constraint="valid JSON string"
        )

    if not isinstance(data, dict):
        raise ValidationError(
            field=field_name,
            value=json_str,
            message="JSON data must be an object",
            constraint="JSON object"
        )

    return data


def validate_list(
    value: List[Any],
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    item_validator: Optional[callable] = None,
) -> List[Any]:
    """
    Validate a list input.

    Args:
        value: List to validate
        field_name: Name of the field being validated
        min_length: Minimum required length
        max_length: Maximum allowed length
        item_validator: Optional validator function for list items

    Returns:
        Validated list

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError(
            field=field_name,
            value=value,
            message="Field must be a list",
            constraint=f"list, min_length={min_length}, max_length={max_length}"
        )

    length = len(value)
    if min_length is not None and length < min_length:
        raise ValidationError(
            field=field_name,
            value=value,
            message=f"List must have at least {min_length} items",
            constraint=f"list, min_length={min_length}"
        )

    if max_length is not None and length > max_length:
        raise ValidationError(
            field=field_name,
            value=value,
            message=f"List must have no more than {max_length} items",
            constraint=f"list, max_length={max_length}"
        )

    # Validate individual items
    if item_validator:
        for i, item in enumerate(value):
            try:
                item_validator(item)
            except ValidationError as e:
                # Include index in the error
                raise ValidationError(
                    field=f"{field_name}[{i}]",
                    value=item,
                    message=e.message,
                    constraint=e.constraint
                )

    return value


def validate_dict(
    value: Dict[str, Any],
    field_name: str,
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, callable]] = None,
) -> Dict[str, Any]:
    """
    Validate a dictionary input.

    Args:
        value: Dictionary to validate
        field_name: Name of the field being validated
        required_fields: List of required field names
        field_validators: Dictionary mapping field names to validator functions

    Returns:
        Validated dictionary

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(
            field=field_name,
            value=value,
            message="Field must be a dictionary",
            constraint="valid dictionary"
        )

    # Check required fields
    if required_fields:
        missing_fields = set(required_fields) - set(value.keys())
        if missing_fields:
            raise ValidationError(
                field=field_name,
                value=value,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                constraint=f"required_fields: {required_fields}"
            )

    # Validate individual fields
    if field_validators:
        for field_name_key, validator in field_validators.items():
            if field_name_key in value:
                try:
                    validator(value[field_name_key])
                except ValidationError as e:
                    # Include field key in the error
                    raise ValidationError(
                        field=f"{field_name}.{field_name_key}",
                        value=value[field_name_key],
                        message=e.message,
                        constraint=e.constraint
                    )

    return value


def validate_url(url: str, field_name: str = "url") -> str:
    """
    Validate a URL.

    Args:
        url: URL to validate
        field_name: Name of the field being validated

    Returns:
        Validated URL string

    Raises:
        ValidationError: If validation fails
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not isinstance(url, str):
        raise ValidationError(
            field=field_name,
            value=url,
            message="URL must be a string",
            constraint="valid URL string"
        )

    if not url_pattern.match(url):
        raise ValidationError(
            field=field_name,
            value=url,
            message="Invalid URL format",
            constraint="valid HTTP/HTTPS URL"
        )

    return url


def validate_email(email: str, field_name: str = "email") -> str:
    """
    Validate an email address.

    Args:
        email: Email address to validate
        field_name: Name of the field being validated

    Returns:
        Validated email string

    Raises:
        ValidationError: If validation fails
    """
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    if not isinstance(email, str):
        raise ValidationError(
            field=field_name,
            value=email,
            message="Email must be a string",
            constraint="valid email address"
        )

    if not email_pattern.match(email):
        raise ValidationError(
            field=field_name,
            value=email,
            message="Invalid email format",
            constraint="valid email address"
        )

    return email


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize a string by removing potentially dangerous characters.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return ""

    # Remove potentially dangerous characters
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)

    # Limit length
    if len(value) > max_length:
        value = value[:max_length]

    return value.strip()


def validate_story_title(title: str) -> str:
    """Validate a story title."""
    return validate_string(
        value=title,
        field_name="title",
        min_length=1,
        max_length=255,
        required=True,
        strip_whitespace=True,
        allowed_chars=(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " ,.!?;:'\"-()[]{}"
            "áéíóúÁÉÍÓÚ"
            "ñÑ"
            "äëïöüÄËÏÖÜ"
        )
    )


def validate_story_setting(setting: str) -> str:
    """Validate a story setting."""
    return validate_string(
        value=setting,
        field_name="setting",
        min_length=10,
        max_length=1000,
        required=True,
        strip_whitespace=True,
        allowed_chars=(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " ,.!?;:'\"-()[]{}"
            "áéíóúÁÉÍÓÚ"
            "ñÑ"
            "äëïöüÄËÏÖÜ"
        )
    )


def validate_character_name(name: str) -> str:
    """Validate a character name."""
    return validate_string(
        value=name,
        field_name="name",
        min_length=1,
        max_length=100,
        required=True,
        strip_whitespace=True,
        allowed_chars=(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " ,.!?;:'\"-()"
        )
    )


def validate_content(content: str, field_name: str = "content", max_length: int = 10000) -> str:
    """Validate story or scene content."""
    return validate_string(
        value=content,
        field_name=field_name,
        min_length=1,
        max_length=max_length,
        required=True,
        strip_whitespace=False,
        allowed_chars=(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " ,.!?;:'\"-()[]{}"
            "\n\r\t"
            "áéíóúÁÉÍÓÚ"
            "ñÑ"
            "äëïöüÄËÏÖÜ"
        )
    )


def validate_date_range(start_date: datetime, end_date: datetime, field_name: str = "date_range") -> Tuple[datetime, datetime]:
    """
    Validate a date range.

    Args:
        start_date: Start date
        end_date: End date
        field_name: Name of the field being validated

    Returns:
        Tuple of (start_date, end_date)

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(start_date, datetime):
        raise ValidationError(
            field=f"{field_name}.start_date",
            value=start_date,
            message="Start date must be a datetime object",
            constraint="datetime object"
        )

    if not isinstance(end_date, datetime):
        raise ValidationError(
            field=f"{field_name}.end_date",
            value=end_date,
            message="End date must be a datetime object",
            constraint="datetime object"
        )

    if start_date > end_date:
        raise ValidationError(
            field=field_name,
            value=(start_date, end_date),
            message="Start date must be before or equal to end date",
            constraint="start_date <= end_date"
        )

    return start_date, end_date