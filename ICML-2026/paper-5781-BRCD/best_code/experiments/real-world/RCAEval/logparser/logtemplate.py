""" To debug regex: https://www.debuggex.com/ """
import json
import re
import toml
import pandas as pd
from collections import OrderedDict

from .utility import mask_dict_values_in_log
from .template import Template


class LogTemplate(Template):
    """
    A class to represent an event type (i.e., log template) for matching events/logs.
    It also offers various useful @staticmethod and @classmethod for mapping/parsing logs.
    """
    verbose = False
    mask_dict = False  # mask dict values in logs

    def __init__(self, id: str = None, template: str = None, known_regex: dict = None):
        if not template:
            raise ValueError("Template cannot be None")
        self.id = id 
        self.template = template
        try:
            self.regex = self._compile_template(template, known_regex)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern in template: {template}. Error: {e}")

    def __repr__(self):
        return f"<{self.id}>_<{self.template}>"

    def _compile_template(self, template: str, known_regex : dict = None) -> re.Pattern:
        """
        Compile the template into a regex pattern.
        """
        # Escape special characters and replace placeholders with regex patterns
        escaped_template = re.escape(template)

        if known_regex is not None:
            for name, pattern in known_regex.items():
                escaped_template = escaped_template.replace(f"<:{name}:>", pattern)

        # Replace <*> with a regex pattern that matches any word
        regex_pattern = escaped_template.replace("<\\*>", ".*?") + "$"
        if self.verbose:
            print(self.id, regex_pattern)
        # Compile the regex pattern
        return re.compile(regex_pattern)

    def is_match(self, event: str) -> bool:
        """
        Check if the event matches the template.
        """
        return bool(self.regex.match(event))
    
    @staticmethod
    def load_templates_from_txt(template_file: str) -> list: 
        """ Load event templates from a txt file.  """
        templates = []
        with open(template_file) as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    templates.append(LogTemplate(template=line))
        return templates

    @staticmethod
    def load_templates_from_toml(template_file):
        """Read templates from toml file"""
        templates = []

        with open(template_file) as f:
            config = toml.load(f)
        regex_patterns = config.get("Regex", {})

        event_types = config.get("LogTemplate", {}) 
        for event_id, template in event_types.items():
            template = LogTemplate(id=event_id, template=template, known_regex=regex_patterns)
            templates.append(template)

        return templates

    @staticmethod 
    def load_templates(template_file):
        if template_file.endswith(".toml"):
            return LogTemplate.load_templates_from_toml(template_file)
        elif template_file.endswith(".txt"):
            return LogTemplate.load_templates_from_txt(template_file)
        else:
            raise ValueError(f"Unsupported template file format: {template_file}. Supported formats are .toml and .txt")

    @staticmethod
    def parse_logs(template_file, log_file):
        """
        Parse logs and match them with templates.
        Args:
            template_file (str): Path to the template file.
            log_file (str): Path to the log file.
        """
        templates = LogTemplate.load_templates(template_file=template_file)
        log_file = open(log_file)
        df = pd.DataFrame(columns=['log', 'event type'])
        for line in log_file:
            line = line.strip()
            if line:
                match = False
                for template in templates:
                    if template.is_match(line):
                        df = df._append({'log': line, 'event type': template.template}, ignore_index=True)
                        match = True
                        break
                if not match:
                    df = df._append({'log': line, 'event type': None}, ignore_index=True)
        log_file.close()
        return df

    @staticmethod
    def is_duplicate(template_file, log_file):
        """
        Check if a log file matches multiple templates.
        """
        templates = LogTemplate.load_templates(template_file)
            
        log_file = open(log_file)
        duplicate = False
        for line in log_file:
            line = line.strip()
            if line:
                matches = []
                for template in templates:
                    if template.is_match(line):
                        matches.append(template.template)
                if len(matches) > 1:
                    duplicate = True
                    print(f"[WARN] Duplicate found!")
                    print(f"log: `{line}`")
                    for match in matches:
                        print(f"template: `{match}`")
        log_file.close()
        if not duplicate:
            print(f"[INFO] No duplicate found.")
        return duplicate
    
    @classmethod
    def is_complete(cls, template_file, log_file, in_progress=False):
        """Check if all logs are match, given a template file"""
        # Load log file and template files
        log_file = open(log_file)
        templates = LogTemplate.load_templates(template_file)

        # Declare local variables
        completeness = True
        match_count = 0
        no_match_count = 0
        no_match_logs = []

        # Loop for each log in log file
        for log in log_file:
            log = log.strip()
            if not log: continue

            if '\\"' in log:
                log = log.encode().decode('unicode_escape').strip()
            log = log.replace("'", '"')

            #if cls.mask_dict:  # in case we wanna keep only the json structure
            masked_log = mask_dict_values_in_log(log)

            match = False

            # for each template
            for template in templates:
                if template.is_match(masked_log): # there is a match
                    match = True
                    match_count += 1
                    break

            # record unmatch logs for reporting
            if not match:
                no_match_logs.append(masked_log)
                no_match_count += 1
                completeness = False

            if in_progress is True and no_match_count == 100:
                print(f"[INFO] Unmatched logs:")
                for log in no_match_logs:
                    print(log)
                return False

        log_file.close()

        if completeness:
            print(f"[INFO] All logs are matched. {match_count} logs matched.")
        else:
            print(f"[INFO] {match_count/(match_count + no_match_count)*100:.2f}% logs matched.")
            print(f"[WARN] {no_match_count} logs unmatched.")
            print(f"[INFO] Unmatched logs:")
            for log in no_match_logs:
                print(log)
        return completeness
