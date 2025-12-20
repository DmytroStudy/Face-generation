import pandas as pd
from imported_notebook import load_generator, generate_faces

def generate (generator):
    male = int(input("Male face (0/1): "))

    generate_faces(generator, male, 1)

if __name__ == "__main__":
    df = pd.read_csv('celeba_reduced.csv', sep=None, engine='python')
    generator = load_generator("W&B/Run_3/generator_3.pth")

    while True: generate(generator)