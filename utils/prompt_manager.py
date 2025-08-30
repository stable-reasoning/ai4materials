import os
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class PromptManager:
    """
    Manages loading and composing prompts using Jinja2 templates
    from a specified local folder.
    """

    def __init__(self, template_folder: str = 'templates'):
        """
        Initializes the PromptManager.

        Args:
            template_folder (str): The path to the folder containing Jinja templates.
                                    Defaults to 'templates'.
        """
        # Ensure the template folder exists
        if not os.path.isdir(template_folder):
            raise FileNotFoundError(f"Template folder not found: {os.path.abspath(template_folder)}")

        self.template_folder = template_folder
        # Set up the Jinja environment to load templates from the specified folder
        self.env = Environment(
            loader=FileSystemLoader(self.template_folder),
            trim_blocks=True,  # Removes the first newline after a block tag
            lstrip_blocks=True,  # Strips leading whitespace from lines with block tags
            autoescape=False  # We are generating text prompts, not HTML
        )
        print(f"PromptManager initialized. Loading templates from: {os.path.abspath(template_folder)}")

    def list_templates(self) -> list[str]:
        return self.env.list_templates()

    def compose_prompt(self, template_name: str, **kwargs) -> str:
        """
        Composes a prompt by rendering a specified Jinja template with given data.

        Args:
            template_name (str): The filename of the template (e.g., 'summarize_code.j2').
            **kwargs: Keyword arguments representing the data to fill into the template.

        Returns:
            str: The rendered prompt string.

        Raises:
            TemplateNotFound: If the specified template_name does not exist.
            Exception: For other potential rendering errors.
        """
        try:
            # Load the template from the environment
            template = self.env.get_template(template_name)
            # Render the template with the provided keyword arguments
            rendered_prompt = template.render(**kwargs)
            return rendered_prompt
        except TemplateNotFound:
            print(f"Error: Template '{template_name}' not found in '{self.template_folder}'.")
            raise
        except Exception as e:
            print(f"Error rendering template '{template_name}': {e}")
            raise
