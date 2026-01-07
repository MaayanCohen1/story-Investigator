"""Tests for StoryParser."""

import pytest

from story_investigator.models import Message
from story_investigator.story_parser import StoryParser


class TestStoryParser:
    """Test suite for StoryParser class."""

    def test_parse_string_single_message(self):
        """Test parsing a single message from XML string."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late.</body>
    </message>
</story>"""
        
        messages = parser.parse_string(xml_content)
        
        assert len(messages) == 1
        assert isinstance(messages[0], Message)
        assert messages[0].sender == "alex"
        assert messages[0].receiver == "six"
        assert messages[0].body == "The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late."
        assert 'sender ref="alex"' in messages[0].original_xml
        assert 'receiver ref="six"' in messages[0].original_xml
        assert messages[0].body in messages[0].original_xml

    def test_parse_string_multiple_messages(self):
        """Test parsing multiple messages from XML string."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
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
        
        messages = parser.parse_string(xml_content)
        
        assert len(messages) == 2
        assert messages[0].sender == "marcus"
        assert messages[0].receiver == "alex"
        assert "USB" in messages[0].body
        assert messages[1].sender == "alex"
        assert messages[1].receiver == "marcus"
        assert "Sure" in messages[1].body
        assert 'sender ref="marcus"' in messages[0].original_xml
        assert 'sender ref="alex"' in messages[1].original_xml

    def test_parse_string_empty_story(self):
        """Test parsing an empty story returns empty list."""
        parser = StoryParser()
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
</story>"""
        
        messages = parser.parse_string(xml_content)
        
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_parse_string_preserves_original_xml(self):
        """Test that original_xml field preserves the complete message XML snippet."""
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
        
        assert len(messages) == 1
        original_xml = messages[0].original_xml
        # Verify all components are in original_xml
        assert "alex" in original_xml
        assert "six" in original_xml
        assert "Test message body" in original_xml
        assert "<message" in original_xml or "<sender" in original_xml
