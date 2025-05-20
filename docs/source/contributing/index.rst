Contributing to SpatioloJI
==========================

Thank you for your interest in contributing to SpatioloJI! We welcome contributions from everyone, regardless of experience level.

Code of Conduct
----------------

This project follows a Code of Conduct to ensure a welcoming experience for everyone. By participating in this project, you agree to abide by its terms.

Ways to Contribute
-------------------

There are many ways to contribute to SpatioloJI:

1. **Bug reports and feature requests**: Submit issues on GitHub
2. **Code contributions**: Submit pull requests with bug fixes or new features
3. **Documentation improvements**: Help improve tutorials, docstrings, and guides
4. **Examples**: Contribute example notebooks showing real-world usage
5. **Testing**: Help improve test coverage or report bugs

Getting Started
-----------------

Setting Up Your Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork locally**:

   .. code-block:: bash

      git clone https://github.com/gynecoloji/SpatioloJI.git
      cd SpatioloJI

3. **Set up a development environment**:

   .. code-block:: bash

      # Create a virtual environment
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      
      # Install development dependencies
      pip install -e ".[dev]"

Development Workflow
---------------------

Making Changes
~~~~~~~~~~~~~~~~~

1. **Create a new branch** for your changes:

   .. code-block:: bash

      git checkout -b feature-or-fix-name

2. **Make your changes** to the code or documentation
3. **Add tests** for any new functionality
4. **Ensure all tests pass**:

   .. code-block:: bash

      pytest

5. **Update documentation** if needed

Code Style
~~~~~~~~~~~~~

SpatioloJI follows these style guidelines:

- **PEP 8** for Python code style
- **NumPy docstring format** for documentation
- **Type hints** for function signatures

We use pre-commit hooks to enforce code style:

.. code-block:: bash

   # Install pre-commit hooks
   pre-commit install
   
   # Run pre-commit on all files
   pre-commit run --all-files

Submitting Changes
~~~~~~~~~~~~~~~~~~~~

1. **Commit your changes** with a descriptive message:

   .. code-block:: bash

      git commit -m "Add feature X" 

2. **Push to your fork**:

   .. code-block:: bash

      git push origin feature-or-fix-name

3. **Create a pull request** from the GitHub interface
4. **Respond to code review feedback** if requested

Documentation
--------------

Good documentation is essential for SpatioloJI. Please follow these guidelines:

Writing Docstrings
~~~~~~~~~~~~~~~~~~~~

Use NumPy-style docstrings for all public functions, classes, and methods:

.. code-block:: python

   def spatial_scatter(adata, color=None, size=None, **kwargs):
       """
       Plot cells in their spatial coordinates.
       
       Parameters
       ----------
       adata : AnnData
           Annotated data matrix with spatial coordinates in `.obsm['spatial']`.
       color : str or None, optional (default: None)
           Key for annotations of observations/cells or a gene name.
       size : float or None, optional (default: None)
           Point size.
       **kwargs
           Additional arguments to pass to `matplotlib.pyplot.scatter`.
           
       Returns
       -------
       matplotlib.axes.Axes
           The axes object with the plot.
           
       Examples
       --------
       >>> import SpatioloJI as sj
       >>> adata = sj.data.example_data()
       >>> sj.pl.spatial_scatter(adata, color='cell_type')
       """

Tutorials and Examples
~~~~~~~~~~~~~~~~~~~~~~~~

Jupyter notebooks make excellent tutorials. When contributing a notebook:

1. Keep the notebook focused on a specific task or workflow
2. Include explanatory text in markdown cells
3. Make sure the notebook runs from start to finish
4. Use relative paths and avoid hardcoded file locations
5. Include visualizations where appropriate

Testing
---------

We use pytest for testing. When adding new features:

Writing Tests
~~~~~~~~~~~~~~~

1. Add tests in the `tests/` directory
2. Ensure good test coverage of your code
3. Test both normal function and edge cases

Running Tests
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run tests with coverage report
   pytest --cov=SpatioloJI

   # Run a specific test file
   pytest tests/test_specific_file.py

Release Process
-----------------

SpatioloJI follows semantic versioning (MAJOR.MINOR.PATCH):

1. MAJOR version for incompatible API changes
2. MINOR version for new functionality in a backward compatible manner
3. PATCH version for backward compatible bug fixes

Getting Help
--------------

If you have questions about contributing to SpatioloJI:

1. Check the documentation
2. Open a discussion on GitHub
3. Contact the maintainers

Thank you for your contributions!