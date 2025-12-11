#!/usr/bin/env python3
"""
Validate conversation log files for corruption and mislabeling.

Checks for:
1. Responses without timestamps (bypassed logger)
2. Suspiciously long responses (>200 chars for Hologram)
3. Claude-style formatting in Hologram responses (*actions*)
4. Duplicate consecutive responses
"""

import re
from pathlib import Path
from typing import List, Tuple


class LogValidator:
    """Validates conversation log files."""
    
    def __init__(self, log_dir: Path = Path("./conversation_logs")):
        """Initialize validator."""
        self.log_dir = log_dir
        self.issues = []
    
    def validate_all_logs(self) -> List[Tuple[str, List[str]]]:
        """
        Validate all log files in directory.
        
        Returns:
            List of (filename, issues) tuples
        """
        results = []
        
        for log_file in sorted(self.log_dir.glob("session_*.log")):
            issues = self.validate_log_file(log_file)
            if issues:
                results.append((log_file.name, issues))
        
        return results
    
    def validate_log_file(self, log_file: Path) -> List[str]:
        """
        Validate a single log file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            List of issue descriptions
        """
        issues = []
        
        try:
            content = log_file.read_text()
        except Exception as e:
            return [f"Could not read file: {e}"]
        
        lines = content.split('\n')
        
        # Check for lines without timestamps
        for i, line in enumerate(lines, 1):
            # Check for speaker prefix without timestamp
            if re.match(r'^(gemini|claude|hologram):', line):
                if not line.startswith('['):
                    issues.append(
                        f"Line {i}: Missing timestamp for speaker: {line[:50]}..."
                    )
        
        # Extract hologram responses
        hologram_responses = []
        current_response = None
        current_line = 0
        
        for i, line in enumerate(lines, 1):
            if re.match(r'\[\d{2}:\d{2}:\d{2}\] hologram:', line):
                if current_response:
                    hologram_responses.append((current_line, current_response))
                # Extract message after timestamp and speaker
                match = re.match(r'\[\d{2}:\d{2}:\d{2}\] hologram: (.+)', line)
                current_response = match.group(1) if match else ""
                current_line = i
            elif current_response and line and not line.startswith('['):
                # Multi-line response
                current_response += ' ' + line
        
        if current_response:
            hologram_responses.append((current_line, current_response))
        
        # Validate hologram responses
        for line_num, response in hologram_responses:
            # Check length
            if len(response) > 200:
                issues.append(
                    f"Line {line_num}: Unusually long Hologram response "
                    f"({len(response)} chars): {response[:50]}..."
                )
            
            # Check for Claude-style formatting
            if '*' in response:
                issues.append(
                    f"Line {line_num}: Hologram response contains '*' formatting "
                    f"(Claude-style): {response[:50]}..."
                )
            
            # Check for conversational markers
            markers = ['you know', 'i think', 'i remember', 'back in', 'when i']
            for marker in markers:
                if marker in response.lower():
                    issues.append(
                        f"Line {line_num}: Hologram response contains conversational "
                        f"marker '{marker}': {response[:50]}..."
                    )
                    break
        
        # Check for duplicate consecutive responses
        for i in range(len(hologram_responses) - 1):
            line1, resp1 = hologram_responses[i]
            line2, resp2 = hologram_responses[i + 1]
            
            # Similar but not identical (suggesting repetition bug)
            if resp1[:50] == resp2[:50] and resp1 != resp2:
                issues.append(
                    f"Lines {line1}-{line2}: Possible duplicate responses: {resp1[:30]}..."
                )
        
        return issues
    
    def print_report(self):
        """Print validation report."""
        results = self.validate_all_logs()
        
        print("=" * 70)
        print("  Conversation Log Validation Report")
        print("=" * 70)
        
        if not results:
            print("\n✓ All log files are valid!")
            return
        
        print(f"\n⚠️  Found issues in {len(results)} log file(s):\n")
        
        for filename, issues in results:
            print(f"\n{filename}:")
            print("-" * 70)
            for issue in issues:
                print(f"  • {issue}")
        
        print("\n" + "=" * 70)
        print(f"Total files checked: {len(list(self.log_dir.glob('session_*.log')))}")
        print(f"Files with issues: {len(results)}")
        print("=" * 70)


def main():
    """Main entry point."""
    validator = LogValidator()
    validator.print_report()


if __name__ == "__main__":
    main()



