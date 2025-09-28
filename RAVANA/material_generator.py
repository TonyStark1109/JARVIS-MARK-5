"""
RAVANA Material Generator Module
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MaterialGenerator(ABC):
    """Abstract base class for material generators."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = {}
    
    @abstractmethod
    def generate(self, **kwargs) -> str:
        """Generate material content."""
        pass

class TextMaterialGenerator(MaterialGenerator):
    """Generates text-based materials."""
    
    def __init__(self):
        super().__init__()
        self.templates = {
            "prompt": "Generate a prompt for: {topic}",
            "instruction": "Follow these instructions: {content}",
            "template": "Use this template: {template}"
        }
    
    def generate(self, material_type: str, **kwargs) -> str:
        """Generate material of specified type."""
        try:
            if material_type not in self.templates:
                raise ValueError(f"Unknown material type: {material_type}")
            
            template = self.templates[material_type]
            result = template.format(**kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Material generation error: {e}")
            return ""

class DataMaterialGenerator(MaterialGenerator):
    """Generates data-based materials."""
    
    def __init__(self):
        super().__init__()
        self.data_types = ['json', 'csv', 'xml', 'yaml']
    
    def generate(self, data_type: str = 'json', structure: Dict[str, Any] = None, count: int = 1) -> str:
        """Generate data material."""
        try:
            if data_type == 'json':
                return self._to_json(structure)
            elif data_type == 'csv':
                return self._to_csv(structure)
            elif data_type == 'xml':
                return self._to_xml(structure)
            else:
                return str(structure)
        except Exception as e:
            self.logger.error(f"Data generation error: {e}")
            return ""
    
    def _to_json(self, data: Dict[str, Any]) -> str:
        """Convert to JSON format."""
        import json
        return json.dumps(data, indent=2)
    
    def _to_csv(self, data: list) -> str:
        """Convert to CSV format."""
        if not data:
            return ""
        
        headers = list(data[0].keys())
        csv_lines = [','.join(headers)]
        
        for item in data:
            row = [str(item.get(header, '')) for header in headers]
            csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)
    
    def _to_xml(self, data: Dict[str, Any]) -> str:
        """Convert to XML format."""
        xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<root>']
        
        for key, value in data.items():
            xml_lines.append(f'  <{key}>{value}</{key}>')
        
        xml_lines.append('</root>')
        return '\n'.join(xml_lines)