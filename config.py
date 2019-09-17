# Dictionary storing network parameters.
params = {
    'latentD': 100, # latent variable dimension
    'batch_size': 128,# Batch size.
    'num_epochs': 100,# Number of epochs to train for.
    'learning_rate': 2e-4,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 25,# After how many epochs to save checkpoints and generate test output.
    'dataset' : 'MNIST',# Dataset to use. Choose from {MNIST, SVHN, CelebA, FashionMNIST}. CASE MUST MATCH EXACTLY!!!!!
    'recon_dir': 'recon', # Reconstruction Directory,
    'dataset_dir': '/home/utiva/workspace/dataset/MNIST',
    'valid_size': 100,
    'save_path': 'save/checkpoint.pth'
}