Release Guide
=============

This describes how to release the package. We basically follow the steps in `Packaging Python Projects <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_.


TestPyPI
--------

We may want to test the distribution before uploading the archives to the PyPI server.
To this end, find the following lines in ``setup.py`` and change the project name to ``appfl-YOUR-NAME`` with a proper version.

.. code-block:: python

    setuptools.setup(
        name="appfl",
        version="0.0.1",
        ...
    )

.. note::

    Make sure to set a higher version if the same version exists in your TestPyPI server. You cannot upload the same version to the server.


Generating distribution archives
++++++++++++++++++++++++++++++++

Let's make sure that ``dist`` directory is empty or does not exist at the root of the repository, where ``pyproject.toml`` is located. Then, running the following command should generate a build distribution ``.whl`` and source archive ``.tar.gz`` files in the ``dist`` directory.

.. code-block:: shell

    python -m build


Uploading the distribution archives
+++++++++++++++++++++++++++++++++++

We upload the distribution archives to the test server by the following command. This requires to have an account for `TestPyPI <https://test.pypi.org>`_.

.. code-block:: shell

    twine upload --repository testpypi dist/*


Installing the new test package
+++++++++++++++++++++++++++++++

Once the distribution archives are uploaded, anyone can install the new test package by the following command:

.. code-block:: shell

    pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple "appfl-YOUR-NAME[dev,examples,analytics]"


Release
-------

If the testing above is successfully, we are ready to release the new package to the PyPI server.
These are key steps to follow:

- Let's not forget to change the project name and set the version properly.
- Generate the distribution archives as above.
- Use ``twine upload dist/*`` to upload the distribution archives.
