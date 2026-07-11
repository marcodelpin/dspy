import re
from typing import Any

from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value, translate_field_type
from dspy.signatures.signature import Signature


def _looks_like_nested_xml(text: str) -> bool:
    """True when the text contains at least one well-formed child tag pair (nested XML)."""
    return re.search(r"<(\w+)>.*?</\1>", text, re.DOTALL) is not None


def _xml_element_to_obj(element):
    children = list(element)
    if not children:
        return (element.text or "").strip()
    tags = [child.tag for child in children]
    if len(set(tags)) == 1 and len(children) > 1:
        # repeated sibling tag -> list
        return [_xml_element_to_obj(child) for child in children]
    obj: dict[str, Any] = {}
    for child in children:
        value = _xml_element_to_obj(child)
        if child.tag in obj:
            if not isinstance(obj[child.tag], list):
                obj[child.tag] = [obj[child.tag]]
            obj[child.tag].append(value)
        else:
            obj[child.tag] = value
    return obj


def _xml_fragment_to_obj(fragment: str):
    """Walk an XML fragment (sibling elements) into a nested dict/list/str structure.

    Distinct sibling tags become a dict, repeated sibling tags become a list, leaf text is stripped.
    Used to accept genuine nested XML for structured fields (#8481).
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(f"<root>{fragment}</root>")
    return _xml_element_to_obj(root)


def _is_structured_annotation(annotation) -> bool:
    """True for pydantic models and list/dict/tuple containers (fields that can hold nested XML)."""
    import typing

    import pydantic

    if typing.get_origin(annotation) in (list, dict, tuple):
        return True
    try:
        return isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel)
    except TypeError:
        return False


class XMLAdapter(ChatAdapter):
    field_pattern = re.compile(r"<(?P<name>\w+)>((?P<content>.*?))</\1>", re.DOTALL)

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            formatted = format_field_value(field_info=field.info, value=field_value)
            output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    def format_field_structure(self, signature: type[Signature]) -> str:
        """
        XMLAdapter requires input and output fields to be wrapped in XML tags like `<field_name>`.
        """

        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        return "\n\n".join(parts).strip()

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]

        messages.append(self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): inputs.get(k)
                for k, v in signature.input_fields.items() if k in inputs
            },
        ))

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        return self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        message = "Respond with the corresponding output fields wrapped in XML tags "
        message += ", then ".join(f"`<{f}>`" for f in signature.output_fields)
        message += "."
        return message

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            if name in signature.output_fields and name not in fields:
                fields[name] = content
        # Cast values using base class parse_value helper
        for k, v in fields.items():
            fields[k] = self._parse_field_value(signature.output_fields[k], v, completion, signature)
        if fields.keys() != signature.output_fields.keys():
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
        return fields

    def _parse_field_value(self, field_info, raw, completion, signature):
        from dspy.adapters.utils import parse_value

        value = raw
        # The write side emits JSON (or a scalar) inside each tag, but an LM may instead emit genuine
        # nested XML for a structured field (e.g. <person><name>..</name><age>..</age></person>). For
        # structured fields, walk that nested XML into a dict/list so parse_value can build the model;
        # scalar fields and the JSON-in-tag form are left untouched (#8481).
        if _is_structured_annotation(field_info.annotation) and _looks_like_nested_xml(raw):
            try:
                value = _xml_fragment_to_obj(raw)
            except Exception:
                value = raw

        try:
            return parse_value(value, field_info.annotation, field_info)
        except Exception as e:
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse field {field_info} with value {raw}: {e}",
            )
