# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

## Development environment setup
In addition to dockerized deployment, a setup is provided using the [Panels](https://panel.holoviz.org/) Python library. This aims to make development easier by isolating the research agent component, enabling it to be tested and tweaked independently from the FastAPI Backend and React UI. 

Follow these steps to get the Panels UI running:

1. Clone the repo

   ```sh
   git clone https://github.com/dankeg/kestrelAI
   ```
2. Install Poetry, if not already installed

   ```sh
   https://python-poetry.org/docs/
   ```
3. Install the project dependencies locally

   ```sh
   poetry install -E agent 
   ```
4. Start the Panels Application

   ```sh
   poetry run panel serve KestrelAI/dashboard.py --autoreload --show
   ```

The UI should automatically launch in the browser.


## Issues and feature requests

You've found a bug in the source code, a mistake in the documentation or maybe you'd like a new feature?Take a look at [GitHub Discussions](https://github.com/dankeg/kestrel/discussions) to see if it's already being discussed.  You can help us by [submitting an issue on GitHub](https://github.com/dankeg/kestrel/issues). Before you create an issue, make sure to search the issue archive -- your issue may have already been addressed!

Please try to create bug reports that are:

- _Reproducible._ Include steps to reproduce the problem.
- _Specific._ Include as much detail as possible: which version, what environment, etc.
- _Unique._ Do not duplicate existing opened issues.
- _Scoped to a Single Bug._ One bug per report.

**Even better: Submit a pull request with a fix or new feature!**

### How to submit a Pull Request

1. Search our repository for open or closed
   [Pull Requests](https://github.com/dankeg/kestrel/pulls)
   that relate to your submission. You don't want to duplicate effort.
2. Fork the project
3. Create your feature branch (`git checkout -b feat/amazing_feature`)
4. Commit your changes (`git commit -m 'feat: add amazing_feature'`) Kestrel uses [conventional commits](https://www.conventionalcommits.org), so please follow the specification in your commit messages.
5. Push to the branch (`git push origin feat/amazing_feature`)
6. [Open a Pull Request](https://github.com/dankeg/kestrel/compare?expand=1)
