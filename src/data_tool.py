import argparse
import cv2
import insightface
import numpy as np
import os
from identity_db import IdentityDB

def main():
    parser = argparse.ArgumentParser(description="Manage CCTV Face Identities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ADD command
    add_parser = subparsers.add_parser("add", help="Add a new face identity from an image")
    add_parser.add_argument("name", help="Name of the person")
    add_parser.add_argument("image_path", help="Path to the image file containing only one face")

    # LIST command
    list_parser = subparsers.add_parser("list", help="List all known identities")

    # DELETE command
    delete_parser = subparsers.add_parser("delete", help="Delete an identity")
    delete_parser.add_argument("name", help="Name of the person to delete")

    args = parser.parse_args()

    db = IdentityDB()

    if args.command == "add":
        if not os.path.exists(args.image_path):
            print(f"Error: Image '{args.image_path}' not found.")
            return

        print("Initializing FaceAnalysis...")
        app = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        img = cv2.imread(args.image_path)
        if img is None:
            print("Error: Could not read image.")
            return

        faces = app.get(img)

        if len(faces) == 0:
            print("Error: No faces found in the image.")
        elif len(faces) > 1:
            print(f"Error: Found {len(faces)} faces. Please provide an image with exactly one face.")
        else:
            embedding = faces[0].embedding
            db.save_identity(args.name, embedding)
            print(f"Successfully added '{args.name}' to the database.")

    elif args.command == "list":
        faces = db.get_known_faces()
        if not faces:
            print("No identities found.")
        else:
            print(f"Found {len(faces)} identities:")
            for name, _ in faces:
                print(f" - {name}")

    elif args.command == "delete":
        if db.delete_identity(args.name):
            print(f"Successfully deleted '{args.name}'.")
        else:
            print(f"Identity '{args.name}' not found.")

if __name__ == "__main__":
    main()
