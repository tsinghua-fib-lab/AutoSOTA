from __future__ import annotations
import os
import copy
import re
from typing import Iterable, List, Mapping, Tuple
from abc import ABC
from collections import defaultdict
from configobj import ConfigObj


class Example(ABC):
    pass

INPUT_PATTERN = re.compile(r'(?:{([^}]+)})')
OUTPUT_PATTERN = re.compile(r'(?:__{([^}]+)}__)')
INPUT_OUTPUT_PATTERN = re.compile('({[^}]+}|__{[^}]+}__)')


class Template:
    """
    Defines a prompting template that can be later filled in with values of actual
    variables for language model prompting. Input variables are denoted with
    `{variable name or python expression}`, and output variables are denoted
    with `__{variable name}__`. After the output has been generated with the
    prompt, this template class can also help extract output variables from
    the generated completion.

    This definition is similar to that in the Demonstrate-Search-Predict paper:
    https://arxiv.org/pdf/2212.14024.pdf
    """
    def __init__(self, template_string : str):
        self.template = self.parse_template_string(template_string)
    
    def __str__(self) -> str:
        """ Construct and return a string representation of the underlying template. """
        template_str = ""
        for elem_dict in self.template:
            if elem_dict['type'] == 'text':
                template_str += elem_dict['value']
            elif elem_dict['type'] == 'input':
                template_str += f"{{{elem_dict['variable_name']}}}"
            elif elem_dict['type'] == 'output':
                template_str += f"__{{{elem_dict['variable_name']}}}__"
            else:
                raise ValueError(f"Cannot recognize type {elem_dict['type']} from template elements.")
        return template_str
    
    def __repr__(self) -> str:
        return f"Template: [{self.__str__()}]"

    def parse_template_string(self, template_string : str) -> List[Mapping]:
        splitted = INPUT_OUTPUT_PATTERN.split(template_string)
        res = []

        def variable_test(expr):
            # Test if the expression is trying to access attributes for a variable
            # named "x", if so, it is a variable read/write expression, otherwise not.
            is_variable = False
            try:
                eval(expr)
            except NameError as ex:
                if str(ex) == "name 'x' is not defined":
                    is_variable = True
            except:
                pass
            return is_variable

        for x in splitted:
            m1 = INPUT_PATTERN.match(x)
            m2 = OUTPUT_PATTERN.match(x)
            is_variable = False
            if m1:
                is_variable = variable_test(m1.group(1))
                if is_variable:
                    res.append({'type': 'input', 'variable_name': m1.group(1)})
            elif m2:
                is_variable = True
                res.append({'type': 'output', 'variable_name': m2.group(1)})

            if not is_variable:
                res.append({'type': 'text', 'value': x})
        return res

    @classmethod
    def render_iterable(cls, xs: Iterable[TemplatedExample]):
        prompt = ""
        rendered_template = []
        if not cls:
            return prompt, rendered_template
        for x1 in xs:
            prompt_, rendered_template_ = x1.render()
            prompt += prompt_
            if len(rendered_template) > 0 and rendered_template[-1]['type'] == 'text' and rendered_template_[0]['type'] == 'text':
                rendered_template[-1]['value'] += rendered_template_[0]['value']
                rendered_template.extend(rendered_template_[1:])
            else:
                rendered_template.extend(rendered_template_)
        return prompt, rendered_template

    def render(self, x : Example) -> Tuple[str, List[Mapping]]:
        prompt = ""
        rendered_template = []
      
        first_output = False
        for component in self.template:
            if component['type'] == 'output':
                first_output = True
                rendered_template.append(copy.copy(component))
            else:
                if component['type'] == 'text':
                    text_update = component['value']
                else:
                    variable = eval(component['variable_name'])
                    if isinstance(variable, TemplatedExample):
                        text_update, rendered_template_ = variable.render()
                        assert len(rendered_template) == 1, "Cannot have output variables in nested templates"
                    elif isinstance(variable, Iterable) and (not variable or isinstance(variable[0], TemplatedExample)):
                        text_update, rendered_template_ = self.render_iterable(variable)
                        assert len(rendered_template) == 1, "Cannot have output variables in nested templates"
                    else:
                        text_update = str(variable)
                if not first_output:
                    prompt += text_update
                if len(rendered_template) > 0 and rendered_template[-1]['type'] == 'text':
                    rendered_template[-1]['value'] += text_update
                else:
                    rendered_template.append({'type': 'text', 'value': text_update})

        return prompt, rendered_template

    @classmethod
    def parse_results(self, rendered_template : List[Mapping], generated_text : str):
        m = None
        included_outputs = len([1 for x in rendered_template if x['type'] == 'output'])

        while not m and included_outputs > 0:
            regex_str = r''
            used_outputs = 0
            for component in rendered_template:
                if used_outputs >= included_outputs: break
                if component['type'] == 'text':
                    regex_str += re.escape(component['value'])
                elif component['type'] == 'output':
                    used_outputs += 1
                    regex_str += '(?P<' + component['variable_name'] + '>.*)'

            m = re.match(regex_str, generated_text, flags=re.DOTALL)

            included_outputs -= 1

        return m.groupdict()
    
    @classmethod
    def from_parsed_template(cls, rendered_template: List[Mapping]):
        template = Template("")
        template.template = rendered_template

        return template


class TemplatedExample(Example):
    """
    An example with a templated associated to help facilitate nested templating.
    For example, a document in open-domain QA can be defined as a TemplatedExample
    with a fixed template, and Template.render will call this example's render
    function to use its properties like (id, text, title) as instructed.
    """
    def __init__(self, template_string : str):
        self.template = Template(template_string)

    def __str__(self) -> str:
        """ Return a string representation of the underlying template. """
        return str(self.template)

    def __repr__(self) -> str:
        return f"TemplatedExample: [{self.__str__()}]"

    def render(self):
        return self.template.render(self)


def template_from_config(config):
    templated_examples = dict()
    dependents = defaultdict(set)
    for name in config:
        template_str = config[name]
        if isinstance(template_str, str) and (re.findall(r'x\.\w+', config[name]) or re.findall(r'getattr\(\s*x\s*,\s*[\'"]\w+', config[name])):
            template = TemplatedExample(template_str)

            for dependency_name in re.findall(r'x\.\w+', config[name]):
                dependents[dependency_name[len('x.'):]].add(name)

            # handle getattr(x, "variable_name") cases
            for dependency_name in re.findall(r'getattr\(\s*x\s*,\s*[\'"]\w+', config[name]):
                m = re.match(r'getattr\(\s*x\s*,\s*[\'"]([^\'"]+)', dependency_name)
                dependents[m.group(1)].add(name)
        else:
            # raw value passed in
            template = template_str
        
        templated_examples[name] = template

    for name in config:
        for dependant_name in dependents[name]:
            setattr(templated_examples[dependant_name], name, templated_examples[name])

    final_prompt = templated_examples['final_prompt']
    return final_prompt


def read_config_from_file_with_inheritance(config_filename, config_dir):
    config = ConfigObj(str(config_dir) + os.path.sep + config_filename)
    if 'base_template' in config:
        base_template = config['base_template']
        base_config = read_config_from_file_with_inheritance(base_template, config_dir)
        del config['base_template']

        base_config.update(config)
        config = base_config

    return config


class ConfigurableTemplatedExample(TemplatedExample):
    """
    A TemplatedExample that can be configured from a config file.

    The config file should at least specify one variable "final_prompt", which
    is the prompt structure used to feed inputs into the LLM. Template config
    files can inherit from other template config files by specifying "base_template"
    as the name of the parent config file, and inheritance/overriding will be
    parsed recursively.

    Unlike Promptus, all variables defined in the config file will be considerd
    "global variables". That is, once defined, they can be used in any other variable
    without worrying about the nested structure of the underlying TemplatedExamples.
    This is a syntactic sugar for simplicity of writing config files.

    This class is deliberately loose in its signature and uses a `kwargs` in its
    constructor to allow for arbitrary example data to be passed in as keyword
    arguments, since these are ultimately defined/used in the config files that
    is only checked at runtime. A pattern to use this class is instantiating it with
    the actual example data before rendering. For instance,

    ```python
    example_data = {
        "response_length": 10,
        "question": "What's going on?",
        "passages": "",
        "history": ""
    }
    template = ConfigurableTemplatedExample("coqa.cfg", **example_data)
    llm_input = template.render()[0]
    ```
    """
    def __init__(self, config_filename, config_dir="template_configs", **kwargs):
        # read configs
        config = read_config_from_file_with_inheritance(config_filename, config_dir)

        config.update(kwargs)

        template = template_from_config(config)

        for key in template.__dict__:
            setattr(self, key, getattr(template, key))
