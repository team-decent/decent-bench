Developer Guide
===============

Wanna contribute to decent-bench? That's great! This guide contains useful information
about development tools, processes, and rules.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~
* `Python 3.13+ <https://www.python.org/downloads/>`_
* `tox <https://pypi.org/project/tox/>`_

Installation for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/team-decent/decent-bench.git
   cd decent-bench
   tox -e dev
   source .tox/dev/bin/activate


Contributing
------------

Commit Messages
~~~~~~~~~~~~~~~
To keep the git history easy to follow and to encourage well-scoped PRs, we follow certain rules for commit messages
when merging into main. Each message uses this template:

.. code-block:: bash
    :caption: Template

    <type>(<scope>): <subject> (#<pr-id>)

    <description>

    closes #<issue-id>

.. code-block:: bash
   :caption: Example

    perf(costs): Cache m_cvx and m_smooth (#105)

    Cache the properties m_cvx and m_smooth where applicable. This led to a
    75% speed up when running ADMM on a logistic regression problem.

    closes #101

Notes:
    - See table below for types.
    - Scope can be a subpackage, module or build tool, e.g. metrics, costs, or sphinx.
    - Max 72 characters per line.
    - Capitalize but do not punctuate subject.
    - Use imperative mood in subject and description.
    - Description explains what changes and why it changes.
    - If the PR has a related issue but doesn't close it, skip the "closes"-keyword and simply reference the issue.

.. list-table::
    :widths: 15 40
    :header-rows: 1
    
    * - Type
      - Description
    * - feat
      - New functionality
    * - perf
      - Performance improvement
    * - ref
      - Refactor
    * - enh
      - Small improvement that doesn't qualify as feat, perf, or ref, e.g. improved variable naming, additional logging,
        or prettier plots
    * - fix
      - Bug fix
    * - test
      - Change to tests
    * - docs
      - Update of comments, docstrings, rst files, or sphinx config
    * - ci
      - CI related change, e.g. modifying GitHub checks or tox environments
    * - license
      - License update

Tooling
~~~~~~~
The dev environment created with :code:`tox -e dev` has all dependencies and dev-dependencies installed. After
activating the dev environment, various tools can be run. Examples include:

.. code-block:: bash

    mypy decent_bench --strict      # find typing issues
    pytest test                     # run tests
    ruff check decent_bench --fix   # find and fix style issues
    ruff format decent_bench        # format code
    make -C docs html               # rebuild html doc files
    tox -e sphinx                   # rebuild rst and html doc files

To ensure that all GitHub status checks pass before opening a PR, simply run :code:`tox` to execute them locally.
To locally verify that doc changes look good, use an html previewer such as
`Live Preview <https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server>`_.


Pull Request Process
~~~~~~~~~~~~~~~~~~~~
1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run :code:`tox` and ensure everything passes.
5. Submit a pull request.
6. Doc changes: click the readthedocs link found in the status checks to verify that everything looks good.


References:
    - https://develop.sentry.dev/engineering-practices/commit-messages/
