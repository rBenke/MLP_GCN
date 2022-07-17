from importlib import import_module

def create_model(model, architecture):
    module = import_module(f"models.{model.lower()}")
    model_class = getattr(module, model.upper())
    model = model_class(architecture)
    return model
