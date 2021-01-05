def get_model(cfg, logger):
    if cfg.arch == 'MBIResPDF':
        from models.p_mbi_pdf import MBIResPDF as Model
        model = Model(logger, quantize=cfg.quantize)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model
