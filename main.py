import os
import argparse
import tensorflow as tf
from gpu_setup import setup_gpu
from data_loader import DataLoader
from embedding_model import make_embedding_model
from siamese_model import SiameseModel
from loss import contrastive_loss

def main(args):
    # Set up GPU environment
    policy = setup_gpu()

    # Load and preprocess data
    data_loader = DataLoader(args.data_dir, args.image_size)
    train_data, test_data = data_loader.load_and_preprocess_data(args.train_frac, args.batch_size)

    # Create embedding model
    embedding_model = make_embedding_model(args.image_size)

    # Create Siamese model
    siamese_model = SiameseModel(embedding_model)
    siamese_model.compile(loss=contrastive_loss(margin=args.margin), optimizer=tf.keras.optimizers.Adam())

    # Train model
    history = siamese_model.train(train_data, test_data, args.epochs)

    # Evaluate model
    siamese_model.evaluate(test_data)

    # Save model
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        siamese_model.save(args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Siamese network for face recognition.")
    parser.add_argument("--data_dir", type=str, default="./data/negative", help="Path to directory containing the face images.")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction of data to use for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--image_size", type=int, default=224, help="Size of input images.")
    parser.add_argument("--margin", type=float, default=1, help="Margin value for contrastive loss.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to directory to save the trained model.")
    args = parser.parse_args()
    main(args)
