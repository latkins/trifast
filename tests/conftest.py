from hypothesis import settings, Phase

settings.register_profile(
    "default", max_examples=20, derandomize=True, phases=[Phase.generate, Phase.target]
)
settings.register_profile(
    "dev", max_examples=100, phases=[Phase.generate, Phase.target, Phase.shrink]
)
settings.register_profile(
    "thorough", max_examples=500, phases=[Phase.generate, Phase.target, Phase.shrink]
)

settings.load_profile("default")
