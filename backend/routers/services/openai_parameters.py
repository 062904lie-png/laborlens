"""Wire parameters for the requested OpenAI families, not answer templates."""


def completion_parameters(model, output_tokens):
    parameters = {'max_completion_tokens': output_tokens}
    if model.startswith(('gpt-4.1', 'gpt-4o')):
        # Non-reasoning models do not accept reasoning_effort.
        return parameters
    parameters['reasoning_effort'] = 'minimal' if model.startswith('gpt-5-nano') else 'none'
    return parameters
