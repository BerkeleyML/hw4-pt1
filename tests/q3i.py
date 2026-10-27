from otter.test_files import test_case

OK_FORMAT = False

name = "q3i"
points = 1

@test_case(points=None, hidden=False)
def test_q3i_sample(env):
    text = env.get('sample_text')
    assert isinstance(text, str) and text.strip(), 'sample_text should be the string returned by generate_sample'
    words = text.split()
    assert words[0] == '<start>', 'sample_text should start with the <start> token that generate_sample prepends to the prompt'
    default_prompt = 'once upon a time'.split()
    prompt_len = len(default_prompt) if words[1:1 + len(default_prompt)] == default_prompt else 0
    assert len(words) > 1 + prompt_len, 'sample_text should contain at least one token generated after the prompt'

