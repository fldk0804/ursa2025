def init_backbone_model(args):
    if args.model == "ViT":
        from models.ViT import VisionTransformer
        return VisionTransformer()
    elif args.model == "CNN":
        from models.CNN import CNNModel
        return CNNModel()
    else:
        raise ValueError(f"Unknown model: {args.model}")
