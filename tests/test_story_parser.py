"""Tests for StoryParser."""

import pytest

from story_investigator.models import Message
from story_investigator.story_parser import StoryParser


class TestStoryParser:
    """Test suite for StoryParser class."""

    @pytest.fixture
    def single_message_xml(self):
        """XML content with a single message."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late.</body>
    </message>
</story>"""

    @pytest.fixture
    def multiple_messages_xml(self):
        """XML content with multiple messages."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="marcus"/>
        <receiver ref="alex"/>
        <body>DM: Bring that USB you borrowed. I need it back tonight. No excuses.</body>
    </message>
    <message id="m2" ts="2025-08-29T17:43:00+10:00">
        <sender ref="alex"/>
        <receiver ref="marcus"/>
        <body>Sure, I'll bring it.</body>
    </message>
</story>"""

    def test_parse_string_single_message_returns_one_message(self, single_message_xml):
        """Test parsing a single message returns exactly one message."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert len(messages) == 1

    def test_parse_string_single_message_returns_message_instance(self, single_message_xml):
        """Test parsing a single message returns Message instance."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert isinstance(messages[0], Message)

    def test_parse_string_single_message_sender(self, single_message_xml):
        """Test parsing a single message extracts correct sender."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert messages[0].sender == "alex"

    def test_parse_string_single_message_receiver(self, single_message_xml):
        """Test parsing a single message extracts correct receiver."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert messages[0].receiver == "six"

    def test_parse_string_single_message_body(self, single_message_xml):
        """Test parsing a single message extracts correct body."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        expected_body = "The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late."
        assert messages[0].body == expected_body

    def test_parse_string_single_message_original_xml_contains_sender(self, single_message_xml):
        """Test parsing a single message preserves sender in original_xml."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert 'sender ref="alex"' in messages[0].original_xml

    def test_parse_string_single_message_original_xml_contains_receiver(self, single_message_xml):
        """Test parsing a single message preserves receiver in original_xml."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert 'receiver ref="six"' in messages[0].original_xml

    def test_parse_string_single_message_original_xml_contains_body(self, single_message_xml):
        """Test parsing a single message preserves body in original_xml."""
        parser = StoryParser()
        messages = parser.parse_string(single_message_xml)
        assert messages[0].body in messages[0].original_xml

    def test_parse_string_multiple_messages_returns_two_messages(self, multiple_messages_xml):
        """Test parsing multiple messages returns correct count."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert len(messages) == 2

    def test_parse_string_multiple_messages_first_sender(self, multiple_messages_xml):
        """Test parsing multiple messages extracts first message sender."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert messages[0].sender == "marcus"

    def test_parse_string_multiple_messages_first_receiver(self, multiple_messages_xml):
        """Test parsing multiple messages extracts first message receiver."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert messages[0].receiver == "alex"

    def test_parse_string_multiple_messages_first_body_contains_usb(self, multiple_messages_xml):
        """Test parsing multiple messages extracts first message body."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert "USB" in messages[0].body

    def test_parse_string_multiple_messages_second_sender(self, multiple_messages_xml):
        """Test parsing multiple messages extracts second message sender."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert messages[1].sender == "alex"

    def test_parse_string_multiple_messages_second_receiver(self, multiple_messages_xml):
        """Test parsing multiple messages extracts second message receiver."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert messages[1].receiver == "marcus"

    def test_parse_string_multiple_messages_second_body_contains_sure(self, multiple_messages_xml):
        """Test parsing multiple messages extracts second message body."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert "Sure" in messages[1].body

    def test_parse_string_multiple_messages_first_original_xml_contains_sender(self, multiple_messages_xml):
        """Test parsing multiple messages preserves first message sender in original_xml."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert 'sender ref="marcus"' in messages[0].original_xml

    def test_parse_string_multiple_messages_second_original_xml_contains_sender(self, multiple_messages_xml):
        """Test parsing multiple messages preserves second message sender in original_xml."""
        parser = StoryParser()
        messages = parser.parse_string(multiple_messages_xml)
        assert 'sender ref="alex"' in messages[1].original_xml

    def test_parse_string_empty_story_returns_list(self):
        """Test parsing an empty story returns a list."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
</story>"""
        messages = parser.parse_string(xml_content)
        assert isinstance(messages, list)

    def test_parse_string_empty_story_returns_empty_list(self):
        """Test parsing an empty story returns empty list."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
</story>"""
        messages = parser.parse_string(xml_content)
        assert len(messages) == 0

    def test_parse_string_preserves_original_xml_contains_sender(self):
        """Test that original_xml field contains sender."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>Test message body</body>
    </message>
</story>"""
        messages = parser.parse_string(xml_content)
        assert "alex" in messages[0].original_xml

    def test_parse_string_preserves_original_xml_contains_receiver(self):
        """Test that original_xml field contains receiver."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>Test message body</body>
    </message>
</story>"""
        messages = parser.parse_string(xml_content)
        assert "six" in messages[0].original_xml

    def test_parse_string_preserves_original_xml_contains_body(self):
        """Test that original_xml field contains body."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>Test message body</body>
    </message>
</story>"""
        messages = parser.parse_string(xml_content)
        assert "Test message body" in messages[0].original_xml

    def test_parse_string_preserves_original_xml_contains_xml_tags(self):
        """Test that original_xml field contains XML tags."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>Test message body</body>
    </message>
</story>"""
        messages = parser.parse_string(xml_content)
        assert "<message" in messages[0].original_xml or "<sender" in messages[0].original_xml
