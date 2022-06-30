class CFG:
    data_dir = "./data"
    image_size = 256
    batch_size = 64
    num_workers = 4

    seed = 1234

    output_dir = "./output/"
    model_dir = output_dir + "models/"
    ckpt_path = model_dir + "best_model-v1.ckpt"

    epochs = 300

    num_workers = 4

    patience = 10
    gpus = 1

    optimizer = "Adam"
    scheduler = "CosineAnnealingLR"
    learning_rate = 5e-5
    T_max = 3
    min_lr = 1e-6
    weight_decay = 1e-6
