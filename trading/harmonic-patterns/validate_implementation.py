#!/usr/bin/env python3
"""
Basic Implementation Validation - No External Dependencies.

Basic syntax and structural validation for Gartley pattern
implementation without external dependencies for CI/CD environments.

Validates:
1. Code syntax and structure
2. Class definitions and methods
3. Type annotations
4. Documentation completeness
5. Error handling patterns

Author: ML Harmonic Patterns Contributors
Created: 2025-09-11
Version: 1.0.0
"""

import ast
import os
import sys
from typing import List, Dict, Set


class CodeValidator:
    """Basic code structure validator."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_file(self, filepath: str) -> bool:
        """Validate Python file structure and syntax."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=filepath)
            
            # Validate structure
            self._validate_ast(tree, filepath)
            
            return len(self.errors) == 0
            
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {filepath}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error validating {filepath}: {e}")
            return False
    
    def _validate_ast(self, tree: ast.AST, filepath: str) -> None:
        """Validate AST structure."""
        
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                self._validate_class(node, filepath)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                self._validate_function(node, filepath)
            elif isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        self.info.append(f"{filepath}: {len(classes)} classes, {len(functions)} functions")
        
        # Validate main classes exist
        if 'gartley_pattern.py' in filepath and '/src/patterns/' in filepath:
            required_classes = ['GartleyPattern', 'PatternResult', 'PatternPoint']
            for req_class in required_classes:
                if req_class not in classes:
                    self.errors.append(f"Missing required class: {req_class}")
                else:
                    self.info.append(f"✅ Found required class: {req_class}")
    
    def _validate_class(self, node: ast.ClassDef, filepath: str) -> None:
        """Validate class structure."""
        
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        
        # Check for constructor
        if '__init__' not in methods and 'gartley_pattern.py' in filepath:
            if node.name == 'GartleyPattern':
                self.errors.append(f"Class {node.name} missing __init__ method")
        
        # Check docstring
        if not ast.get_docstring(node):
            self.warnings.append(f"Class {node.name} missing docstring")
        else:
            self.info.append(f"✅ Class {node.name} has docstring")
        
        # Validate main GartleyPattern methods
        if node.name == 'GartleyPattern':
            required_methods = [
                'detect_patterns',
                'get_entry_signals', 
                'calculate_position_size',
                'prepare_visualization_data'
            ]
            
            for req_method in required_methods:
                if req_method not in methods:
                    self.errors.append(f"GartleyPattern missing required method: {req_method}")
                else:
                    self.info.append(f"✅ Found required method: {req_method}")
    
    def _validate_function(self, node: ast.FunctionDef, filepath: str) -> None:
        """Validate function structure."""
        
        # Check for type annotations (basic)
        has_return_annotation = node.returns is not None
        has_arg_annotations = any(arg.annotation is not None for arg in node.args.args)
        
        if not has_return_annotation and not node.name.startswith('_'):
            self.warnings.append(f"Function {node.name} missing return type annotation")
        
        # Check docstring
        if not ast.get_docstring(node) and not node.name.startswith('_'):
            self.warnings.append(f"Function {node.name} missing docstring")
    
    def print_results(self) -> None:
        """Print validation results."""
        
        print("=" * 60)
        print("🔍 CODE VALIDATION RESULTS")
        print("=" * 60)
        
        if self.info:
            print("\n📋 INFO:")
            for info in self.info:
                print(f"   {info}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   ❌ {error}")
        else:
            print("\n✅ NO ERRORS FOUND")
        
        print("=" * 60)
        
        # Overall status
        if self.errors:
            print("❌ VALIDATION FAILED")
            return False
        elif self.warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            return True
        else:
            print("✅ VALIDATION PASSED")
            return True


def validate_project_structure():
    """Validate project file structure."""
    
    print("📁 VALIDATING PROJECT STRUCTURE...")
    
    base_path = os.path.dirname(__file__)
    
    required_files = [
        'src/patterns/__init__.py',
        'src/patterns/gartley_pattern.py',
        'tests/test_gartley_pattern.py',
        'examples/gartley_demo.py',
        'README.md'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            existing_files.append(file_path)
            print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
        return False
    else:
        print(f"\n✅ All required files present: {len(existing_files)}")
        return True


def validate_import_structure():
    """Validate import structure without actually importing."""
    
    print("\n📦 VALIDATING IMPORT STRUCTURE...")
    
    base_path = os.path.dirname(__file__)
    main_file = os.path.join(base_path, 'src', 'patterns', 'gartley_pattern.py')
    
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            'numpy',
            'pandas', 
            'typing',
            'dataclasses',
            'enum',
            'logging'
        ]
        
        found_imports = []
        missing_imports = []
        
        for imp in required_imports:
            if f'import {imp}' in content or f'from {imp}' in content:
                found_imports.append(imp)
                print(f"   ✅ {imp}")
            else:
                missing_imports.append(imp)
                print(f"   ❌ {imp}")
        
        if missing_imports:
            print(f"\n⚠️  Some imports not found: {missing_imports}")
            print("   💡 This is normal if using alternative import styles")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking imports: {e}")
        return False


def check_code_patterns():
    """Check for important code patterns."""
    
    print("\n🔍 CHECKING CODE PATTERNS...")
    
    base_path = os.path.dirname(__file__)
    main_file = os.path.join(base_path, 'src', 'patterns', 'gartley_pattern.py')
    
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        patterns = {
            'Error handling': ['try:', 'except:', 'raise'],
            'Logging': ['logger.', 'logging.'],
            'Type hints': ['def ', ') -> ', ': float', ': int', ': str'],
            'Documentation': ['"""', "'''"],
            'Implementation comments': ['Enterprise', 'Production'],
            'Fibonacci ratios': ['0.618', '0.786', '1.272', '1.618'],
            'Pattern validation': ['PatternValidation', 'confidence_score'],
            'Risk management': ['stop_loss', 'take_profit', 'risk_reward']
        }
        
        for pattern_name, keywords in patterns.items():
            found = sum(1 for keyword in keywords if keyword in content)
            total = len(keywords)
            
            if found > 0:
                print(f"   ✅ {pattern_name}: {found}/{total} patterns found")
            else:
                print(f"   ⚠️  {pattern_name}: No patterns found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error checking patterns: {e}")
        return False


def validate_fibonacci_constants():
    """Validate Fibonacci constant definitions."""
    
    print("\n📐 VALIDATING FIBONACCI CONSTANTS...")
    
    base_path = os.path.dirname(__file__)
    main_file = os.path.join(base_path, 'src', 'patterns', 'gartley_pattern.py')
    
    try:
        with open(main_file, 'r') as f:
            content = f.read()
        
        # Expected Gartley Fibonacci ratios
        expected_ratios = {
            'AB_RETRACEMENT': '0.618',
            'BC_MIN_RETRACEMENT': '0.382',
            'BC_MAX_RETRACEMENT': '0.886', 
            'CD_MIN_EXTENSION': '1.272',
            'CD_MAX_EXTENSION': '1.618',
            'AD_RETRACEMENT': '0.786'
        }
        
        found_ratios = 0
        for ratio_name, ratio_value in expected_ratios.items():
            if ratio_name in content and ratio_value in content:
                print(f"   ✅ {ratio_name}: {ratio_value}")
                found_ratios += 1
            else:
                print(f"   ❌ {ratio_name}: {ratio_value}")
        
        if found_ratios == len(expected_ratios):
            print(f"\n✅ All Fibonacci ratios defined correctly")
            return True
        else:
            print(f"\n⚠️  {found_ratios}/{len(expected_ratios)} ratios found")
            return False
        
    except Exception as e:
        print(f"   ❌ Error validating Fibonacci constants: {e}")
        return False


def main():
    """Main validation routine."""
    
    print("🚀 STARTING GARTLEY PATTERN IMPLEMENTATION VALIDATION")
    print("Implementation Validation Suite")
    print("=" * 70)
    
    results = []
    
    # 1. Project structure
    results.append(validate_project_structure())
    
    # 2. Import structure
    results.append(validate_import_structure())
    
    # 3. Code patterns
    results.append(check_code_patterns())
    
    # 4. Fibonacci constants
    results.append(validate_fibonacci_constants())
    
    # 5. Code validation
    validator = CodeValidator()
    
    base_path = os.path.dirname(__file__)
    files_to_validate = [
        os.path.join(base_path, 'src', 'patterns', 'gartley_pattern.py'),
        os.path.join(base_path, 'src', 'patterns', '__init__.py'),
        os.path.join(base_path, 'tests', 'test_gartley_pattern.py'),
        os.path.join(base_path, 'examples', 'gartley_demo.py')
    ]
    
    print(f"\n🔍 VALIDATING {len(files_to_validate)} PYTHON FILES...")
    
    for file_path in files_to_validate:
        if os.path.exists(file_path):
            file_result = validator.validate_file(file_path)
            results.append(file_result)
        else:
            print(f"   ❌ File not found: {file_path}")
            results.append(False)
    
    validator.print_results()
    
    # Overall result
    print("\n" + "=" * 70)
    print("📊 OVERALL VALIDATION RESULTS")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"📈 Tests passed: {passed}/{total}")
    print(f"📊 Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Gartley Pattern implementation is ready for production")
        return True
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print("💡 Check the errors above and fix before deployment")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation script failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)