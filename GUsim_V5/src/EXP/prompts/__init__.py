import os


def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


class PromptTemplate:
    """A prompt template."""

    def __init__(self, template: str, in_context_example: str = "", additional_args=None):
        if additional_args is None:
            additional_args = {}
        self.template: str = template
        self.in_context_example = in_context_example
        # if more args are needed, add them here,it will be passed to the template
        self.additional_args = additional_args

    def __call__(self, **kwargs) -> str:
        return self.template.format(**kwargs)
