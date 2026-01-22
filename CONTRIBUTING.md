# Contributing to Project Lighthouse Anonymize

Thank you for your interest in contributing!

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/project-lighthouse-anonymize.git
cd project-lighthouse-anonymize

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Making Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Add/update tests (maintain 90% coverage)
4. Run tests and linting: `ruff format src/ tests/ && ruff check src/ tests/ && pytest`
5. Update documentation if needed
6. Commit with clear messages

## Pull Request Process

1. Update README.md if needed
2. Update CHANGELOG.md with your changes
3. Ensure all tests pass
4. Fill out the pull request template completely
5. Request review from maintainers

## Testing

- Maintain 90% test coverage
- Include both unit tests and integration tests
- Use descriptive test names: `test_k_anonymize_with_invalid_k_raises_error`
- **Test Style**: Use class-based tests (pytest class style) for consistency
  - Group related tests in test classes (e.g., `TestKAnonymize`)
  - Use `@staticmethod` for test helper methods that don't need instance state
  - Use `@property` or class attributes for shared test data when appropriate

## Documentation

- Use NumPy-style docstrings
- Update relevant documentation in `docs/`
- Include code examples where appropriate

## Questions?

Open an issue or discussion on GitHub.
