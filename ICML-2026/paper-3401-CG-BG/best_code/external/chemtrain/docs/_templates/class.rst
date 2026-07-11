{{ ":modulename:`%s`" | format(fullname | replace("chemtrain.", "")) | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: _autosummary
      :template: method.rst

   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
      :annotation:
   {%- endfor %}
   {% endif %}
   {% endblock %}