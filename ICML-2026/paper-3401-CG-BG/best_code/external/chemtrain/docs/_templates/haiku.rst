{{ ":modulename:`%s`" | format(fullname | replace("chemtrain.", "")) | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. automethod:: __init__
   .. automethod:: __call__

   .. autosummary::
      :toctree: _autosummary
      :template: method.rst

   {% for item in methods %}
   {% if item != '__init__' %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}