from loaders.multimodalcardiac import MultiModalCardiacLoader as MultiModalCardiacLoader



def init_loader(dataset):
    """
    Factory method for initialising data loaders by name.
    """
    if dataset == 'multimodalcardiac':
        return MultiModalCardiacLoader()
    return None