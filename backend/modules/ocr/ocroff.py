#!/usr/bin/env python3
"""
JARVIS Mark 5 - OCR (Optical Character Recognition) Module
Extracts text from images using EasyOCR and Tesseract
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Union
import io
import base64

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import easyocr
    import cv2
    import numpy as np
    from PIL import Image
    EASYOCR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"EasyOCR not available: {e}")
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Tesseract not available: {e}")
    TESSERACT_AVAILABLE = False

class OCRProcessor:
    """OCR processor using EasyOCR and Tesseract"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.easyocr_reader = None
        self.tesseract_available = TESSERACT_AVAILABLE
        
        # Initialize EasyOCR if available
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                self.logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
        
        self.logger.info(f"OCR Processor initialized - EasyOCR: {EASYOCR_AVAILABLE}, Tesseract: {TESSERACT_AVAILABLE}")
    
    def extract_text_easyocr(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        try:
            if not self.easyocr_reader:
                return {"success": False, "error": "EasyOCR not available", "text": ""}
            
            # Convert image to numpy array if needed
            if isinstance(image_input, str):
                # File path
                image = cv2.imread(image_input)
                if image is None:
                    return {"success": False, "error": "Could not load image", "text": ""}
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = np.array(image_input)
            else:
                # Already numpy array
                image = image_input
            
            # Perform OCR
            results = self.easyocr_reader.readtext(image)
            
            # Extract text and confidence scores
            extracted_text = []
            confidence_scores = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    extracted_text.append(text)
                    confidence_scores.append(confidence)
            
            full_text = " ".join(extracted_text)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return {
                "success": True,
                "text": full_text,
                "confidence": avg_confidence,
                "method": "EasyOCR",
                "details": results
            }
            
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {e}")
            return {"success": False, "error": str(e), "text": ""}
    
    def extract_text_tesseract(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Extract text using Tesseract"""
        try:
            if not self.tesseract_available:
                return {"success": False, "error": "Tesseract not available", "text": ""}
            
            # Convert image to PIL Image if needed
            if isinstance(image_input, str):
                # File path
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                # OpenCV image
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                # Already PIL Image
                image = image_input
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": avg_confidence / 100.0,  # Convert to 0-1 scale
                "method": "Tesseract",
                "details": data
            }
            
        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return {"success": False, "error": str(e), "text": ""}
    
    def extract_text_hybrid(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Extract text using both methods and return the best result"""
        try:
            results = []
            
            # Try EasyOCR first
            if self.easyocr_reader:
                easyocr_result = self.extract_text_easyocr(image_input)
                if easyocr_result["success"]:
                    results.append(easyocr_result)
            
            # Try Tesseract
            if self.tesseract_available:
                tesseract_result = self.extract_text_tesseract(image_input)
                if tesseract_result["success"]:
                    results.append(tesseract_result)
            
            if not results:
                return {"success": False, "error": "No OCR methods available", "text": ""}
            
            # Return the result with highest confidence
            best_result = max(results, key=lambda x: x.get("confidence", 0))
            
            return {
                "success": True,
                "text": best_result["text"],
                "confidence": best_result["confidence"],
                "method": f"Hybrid (best: {best_result['method']})",
                "all_results": results
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid OCR extraction failed: {e}")
            return {"success": False, "error": str(e), "text": ""}
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
            elif isinstance(image_input, Image.Image):
                image = np.array(image_input)
            else:
                image = image_input
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply threshold
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image_input if isinstance(image_input, np.ndarray) else np.array(image_input)

# Global OCR processor instance
_ocr_processor = None

def get_ocr_processor() -> OCRProcessor:
    """Get or create OCR processor instance"""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor

def ocr_off(image_input: Union[str, np.ndarray, Image.Image], method: str = "hybrid") -> str:
    """
    Main OCR function - extracts text from image
    
    Args:
        image_input: Image file path, numpy array, or PIL Image
        method: OCR method ("easyocr", "tesseract", or "hybrid")
    
    Returns:
        Extracted text as string
    """
    try:
        processor = get_ocr_processor()
        
        # Preprocess image for better results
        processed_image = processor.preprocess_image(image_input)
        
        # Choose extraction method
        if method.lower() == "easyocr":
            result = processor.extract_text_easyocr(processed_image)
        elif method.lower() == "tesseract":
            result = processor.extract_text_tesseract(processed_image)
        else:  # hybrid
            result = processor.extract_text_hybrid(processed_image)
        
        if result["success"]:
            return result["text"]
        else:
            logger.error(f"OCR failed: {result.get('error', 'Unknown error')}")
            return f"OCR Error: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"OCR function error: {e}")
        return f"OCR Error: {str(e)}"

def ocr_detailed(image_input: Union[str, np.ndarray, Image.Image], method: str = "hybrid") -> Dict[str, Any]:
    """
    Detailed OCR function - returns full result with metadata
    
    Args:
        image_input: Image file path, numpy array, or PIL Image
        method: OCR method ("easyocr", "tesseract", or "hybrid")
    
    Returns:
        Dictionary with text, confidence, method, and details
    """
    try:
        processor = get_ocr_processor()
        
        # Preprocess image for better results
        processed_image = processor.preprocess_image(image_input)
        
        # Choose extraction method
        if method.lower() == "easyocr":
            result = processor.extract_text_easyocr(processed_image)
        elif method.lower() == "tesseract":
            result = processor.extract_text_tesseract(processed_image)
        else:  # hybrid
            result = processor.extract_text_hybrid(processed_image)
        
        return result
            
    except Exception as e:
        logger.error(f"Detailed OCR function error: {e}")
        return {"success": False, "error": str(e), "text": ""}

# Example usage and testing
def main():
    """Test OCR functionality"""
    print("🔍 JARVIS OCR Module Test")
    print("=" * 40)
    
    # Test with a sample image (you would provide an actual image path)
    test_image_path = "test_image.png"  # Replace with actual image path
    
    if os.path.exists(test_image_path):
        print(f"Testing OCR with image: {test_image_path}")
        
        # Test basic OCR
        text = ocr_off(test_image_path)
        print(f"Extracted text: {text}")
        
        # Test detailed OCR
        result = ocr_detailed(test_image_path)
        print(f"Detailed result: {result}")
    else:
        print("No test image found. Please provide an image to test OCR functionality.")
        print("Available OCR methods:")
        print(f"  - EasyOCR: {EASYOCR_AVAILABLE}")
        print(f"  - Tesseract: {TESSERACT_AVAILABLE}")

if __name__ == "__main__":
    main()
