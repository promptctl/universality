import pytest

from uni.template import SLOT, TemplateError, load_templates, parse_template


def test_render_puts_the_state_in_the_slot():
    assert parse_template("t", "Say:\n{state}!").render("hi") == "Say:\nhi!"


def test_a_state_holding_the_slot_is_sent_as_written():
    assert parse_template("t", "Say: {state}").render(SLOT) == "Say: {state}"


def test_text_round_trips():
    assert parse_template("t", "a {state} b").text == "a {state} b"


@pytest.mark.parametrize("text, count", [("no slot", 0), ("{state} and {state}", 2)])
def test_a_template_must_hold_the_slot_exactly_once(text, count):
    with pytest.raises(TemplateError, match=f"found it {count} times"):
        parse_template("t", text)


def test_a_template_must_be_a_string():
    with pytest.raises(TemplateError, match="must be a string"):
        parse_template("t", 3)


def test_the_shipped_templates_parse():
    templates = load_templates()
    assert {"identity", "empty", "rewrite", "summarize"} <= templates.keys()
    assert templates["empty"].render("anything") == "anything"
    assert templates["empty"].render("") == ""
