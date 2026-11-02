"""AnuLM: a small trainable language model."""
__version__ = "0.1.0"
__all__ = ["AnuLM", "AnuLMConfig", "load_checkpoint"]

def __getattr__(name):
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
